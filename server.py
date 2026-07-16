from __future__ import annotations

import argparse
import base64
import binascii
import json
import mimetypes
import socket
import subprocess
import traceback
import uuid
from http import HTTPStatus
from http.cookies import SimpleCookie
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from database.store import (
    DuplicateUserError,
    create_bookmark,
    create_session,
    create_user,
    delete_bookmark,
    delete_session,
    get_user_by_session,
    init_db,
    list_bookmarks,
    authenticate_user,
    SESSION_SECONDS,
    update_bookmark_log_text,
)


ROOT = Path(__file__).resolve().parent
SESSION_COOKIE = "ai_video_session"
RECORDINGS_DIR = ROOT / "recordings"
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_CHUNK_BYTES = 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".webm", ".mp4"}


class AppHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def allowed_cors_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        if not origin:
            return None
        allowed_prefixes = (
            "http://localhost:",
            "http://127.0.0.1:",
            "https://localhost:",
            "https://127.0.0.1:",
        )
        return origin if origin.startswith(allowed_prefixes) else None

    def end_headers(self) -> None:
        origin = self.allowed_cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Video-Extension, X-Chunk-Index, X-Chunk-Total, X-Chunk-Offset"
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/":
            self.redirect("/index.html")
            return

        if path == "/logout":
            delete_session(self.session_token())
            self.redirect("/index.html", clear_cookie=True)
            return

        if path == "/api/me":
            user = self.current_user()
            if not user:
                self.send_json({"error": "Not logged in."}, HTTPStatus.UNAUTHORIZED)
                return
            self.send_json({"user": user})
            return

        if path == "/api/bookmarks":
            user = self.require_user()
            if not user:
                return

            bookmarks = [
                self.with_recording_url(bookmark)
                for bookmark in list_bookmarks(user["id"])
            ]
            self.send_json({"bookmarks": bookmarks})
            return

        if path in {"/main.html", "/savedLog.html"} and not self.current_user():
            self.redirect("/login.html")
            return

        # This also serves /recordings/bookmark_<id>.webm from ROOT/recordings.
        super().do_GET()

    def do_POST(self) -> None:
        try:
            self._do_POST()
        except Exception as exc:
            print(f"Unhandled POST error for {self.path}: {exc}", flush=True)
            traceback.print_exc()
            try:
                self.send_json(
                    {"error": f"Server error while handling upload: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except Exception:
                pass

    def _do_POST(self) -> None:
        path = urlparse(self.path).path

        if path == "/signup":
            self.handle_signup()
            return

        if path == "/login":
            self.handle_login()
            return

        if path == "/api/bookmarks":
            self.handle_create_bookmark()
            return

        if path == "/api/youtube/download":
            self.handle_youtube_download()
            return

        if path.startswith("/api/bookmarks/") and path.endswith("/recording/chunk"):
            self.handle_upload_recording_chunk(path)
            return

        if path.startswith("/api/bookmarks/") and path.endswith("/recording"):
            self.handle_upload_recording(path)
            return

        if path.startswith("/api/bookmarks/"):
            self.handle_update_bookmark_log(path)
            return

        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_PATCH(self) -> None:
        path = urlparse(self.path).path

        if path.startswith("/api/bookmarks/"):
            self.handle_update_bookmark_log(path)
            return

        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path

        if path.startswith("/api/bookmarks/"):
            user = self.require_user()
            if not user:
                return

            try:
                bookmark_id = int(path.rsplit("/", 1)[1])
            except ValueError:
                self.send_json({"error": "Invalid bookmark id."}, HTTPStatus.BAD_REQUEST)
                return

            if not delete_bookmark(user["id"], bookmark_id):
                self.send_json({"error": "Bookmark not found."}, HTTPStatus.NOT_FOUND)
                return

            self.delete_recording(bookmark_id)
            self.send_json({"deleted": True})
            return

        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def handle_signup(self) -> None:
        data = self.read_request_data()
        username = data.get("username", "").strip()
        password = data.get("password", "")
        confirm_password = data.get("confirm-password", "")

        if not username or not password:
            self.redirect("/signup.html?error=missing")
            return

        if password != confirm_password:
            self.redirect("/signup.html?error=mismatch")
            return

        try:
            user = create_user(username, password)
        except DuplicateUserError:
            self.redirect("/signup.html?error=exists")
            return
        except ValueError:
            self.redirect("/signup.html?error=missing")
            return

        token = create_session(user["id"])
        self.redirect("/main.html", session_token=token)

    def handle_login(self) -> None:
        data = self.read_request_data()
        user = authenticate_user(data.get("username", ""), data.get("password", ""))

        if not user:
            self.redirect("/login.html?error=invalid")
            return

        token = create_session(user["id"])
        self.redirect("/main.html", session_token=token)

    def handle_create_bookmark(self) -> None:
        user = self.require_user()
        if not user:
            return

        content_type = self.headers.get("Content-Type", "")
        video_upload = None

        try:
            if "multipart/form-data" in content_type:
                data, video_upload = self.read_multipart_data()
            else:
                data = self.read_request_data()
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        try:
            bookmark = create_bookmark(
                user_id=user["id"],
                video_id=data.get("video_id"),
                video_title=data.get("video_title"),
                log_text=data.get("log_text", ""),
            )
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        try:
            if video_upload is not None:
                self.save_recording(bookmark["id"], video_upload)
        except ValueError as exc:
            # Do not leave a log-only bookmark if its requested recording failed.
            delete_bookmark(user["id"], bookmark["id"])
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except OSError as exc:
            delete_bookmark(user["id"], bookmark["id"])
            self.send_json(
                {"error": f"Unable to save camera recording: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        bookmark = self.with_recording_url(bookmark)
        self.send_json({"bookmark": bookmark}, HTTPStatus.CREATED)

    def handle_upload_recording_chunk(self, path: str) -> None:
        user = self.require_user()
        if not user:
            return

        parts = [part for part in path.split("/") if part]
        if (
            len(parts) != 5
            or parts[0] != "api"
            or parts[1] != "bookmarks"
            or parts[3] != "recording"
            or parts[4] != "chunk"
        ):
            self.send_json({"error": "Invalid recording chunk URL."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            bookmark_id = int(parts[2])
            chunk_index = int(self.headers.get("X-Chunk-Index", "-1"))
            chunk_total = int(self.headers.get("X-Chunk-Total", "0"))
            chunk_offset = int(self.headers.get("X-Chunk-Offset", "-1"))
        except ValueError:
            self.send_json({"error": "Invalid recording chunk metadata."}, HTTPStatus.BAD_REQUEST)
            return

        owned_ids = {int(item["id"]) for item in list_bookmarks(user["id"])}
        if bookmark_id not in owned_ids:
            self.send_json({"error": "Bookmark not found."}, HTTPStatus.NOT_FOUND)
            return

        if chunk_total <= 0 or chunk_total > 10000 or chunk_index < 0 or chunk_index >= chunk_total:
            self.send_json({"error": "Invalid recording chunk number."}, HTTPStatus.BAD_REQUEST)
            return
        if chunk_offset < 0:
            self.send_json({"error": "Invalid recording chunk offset."}, HTTPStatus.BAD_REQUEST)
            return

        extension = str(self.headers.get("X-Video-Extension", "webm")).strip().lower()
        if extension not in {"webm", "mp4"}:
            extension = "webm"

        try:
            chunk = self.read_body_bytes(MAX_CHUNK_BYTES)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        except (OSError, socket.timeout) as exc:
            self.send_json({"error": f"Recording chunk connection was interrupted: {exc}"}, HTTPStatus.BAD_REQUEST)
            return

        if not chunk:
            self.send_json({"error": "The recording chunk is empty."}, HTTPStatus.BAD_REQUEST)
            return

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = RECORDINGS_DIR / f".bookmark_{bookmark_id}.{extension}.part"
        final_path = RECORDINGS_DIR / f"bookmark_{bookmark_id}.{extension}"

        if chunk_index == 0:
            self.delete_recording(bookmark_id)
            temp_path.unlink(missing_ok=True)

        current_size = temp_path.stat().st_size if temp_path.exists() else 0
        if current_size != chunk_offset:
            self.send_json(
                {
                    "error": (
                        f"Recording chunk is out of order. "
                        f"Expected offset {current_size}, received {chunk_offset}."
                    )
                },
                HTTPStatus.CONFLICT,
            )
            return

        try:
            with temp_path.open("ab") as output:
                output.write(chunk)
        except OSError as exc:
            self.send_json({"error": f"Unable to save recording chunk: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        completed = chunk_index == chunk_total - 1
        if completed:
            final_path.unlink(missing_ok=True)
            temp_path.replace(final_path)

        print(
            f"Recording chunk saved: bookmark={bookmark_id}, "
            f"chunk={chunk_index + 1}/{chunk_total}, bytes={len(chunk)}, completed={completed}",
            flush=True,
        )

        self.send_json(
            {
                "bookmark_id": bookmark_id,
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
                "completed": completed,
                "video_url": f"/recordings/{final_path.name}" if completed else None,
            },
            HTTPStatus.CREATED,
        )

    def handle_upload_recording(self, path: str) -> None:
        user = self.require_user()
        if not user:
            return

        parts = [part for part in path.split("/") if part]
        if len(parts) != 4 or parts[0] != "api" or parts[1] != "bookmarks" or parts[3] != "recording":
            self.send_json({"error": "Invalid recording upload URL."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            bookmark_id = int(parts[2])
        except ValueError:
            self.send_json({"error": "Invalid bookmark id."}, HTTPStatus.BAD_REQUEST)
            return

        owned_ids = {int(item["id"]) for item in list_bookmarks(user["id"])}
        if bookmark_id not in owned_ids:
            self.send_json({"error": "Bookmark not found."}, HTTPStatus.NOT_FOUND)
            return

        content_type = str(self.headers.get("Content-Type", "")).lower()
        transfer_encoding = str(self.headers.get("Transfer-Encoding", "")).lower()
        content_length = self.headers.get("Content-Length", "unknown")
        print(
            f"Recording upload started: bookmark={bookmark_id}, "
            f"content_type={content_type}, content_length={content_length}, "
            f"transfer_encoding={transfer_encoding or 'none'}",
            flush=True,
        )

        try:
            raw_body = self.read_body_bytes(MAX_UPLOAD_BYTES * 2)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        except (OSError, socket.timeout) as exc:
            self.send_json(
                {"error": f"The recording upload connection was interrupted: {exc}"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        requested_extension = str(self.headers.get("X-Video-Extension", "")).strip().lower()
        detected_content_type = content_type

        if "application/json" in content_type:
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.send_json({"error": "Invalid recording JSON body."}, HTTPStatus.BAD_REQUEST)
                return

            requested_extension = str(payload.get("extension", requested_extension)).strip().lower()
            detected_content_type = str(payload.get("content_type", "video/webm")).lower()
            encoded_video = str(payload.get("video_base64", "")).strip()
            if not encoded_video:
                self.send_json({"error": "The camera recording is empty."}, HTTPStatus.BAD_REQUEST)
                return

            try:
                video_bytes = base64.b64decode(encoded_video, validate=True)
            except (ValueError, binascii.Error):
                self.send_json({"error": "Invalid base64 camera recording."}, HTTPStatus.BAD_REQUEST)
                return
        else:
            video_bytes = raw_body

        if not video_bytes:
            self.send_json({"error": "The camera recording is empty."}, HTTPStatus.BAD_REQUEST)
            return
        if len(video_bytes) > MAX_UPLOAD_BYTES:
            self.send_json(
                {"error": "The camera recording is too large. Maximum size is 250 MB."},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        if requested_extension not in {"webm", "mp4"}:
            requested_extension = "mp4" if "mp4" in detected_content_type else "webm"

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.delete_recording(bookmark_id)
        destination = RECORDINGS_DIR / f"bookmark_{bookmark_id}.{requested_extension}"

        try:
            destination.write_bytes(video_bytes)
        except OSError as exc:
            destination.unlink(missing_ok=True)
            self.send_json(
                {"error": f"Unable to save camera recording: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        print(
            f"Recording upload completed: bookmark={bookmark_id}, "
            f"bytes={destination.stat().st_size}, file={destination.name}",
            flush=True,
        )

        bookmark = next(
            item for item in list_bookmarks(user["id"])
            if int(item["id"]) == bookmark_id
        )
        result = self.with_recording_url(bookmark)
        result["recording_size"] = destination.stat().st_size
        self.send_json({"bookmark": result}, HTTPStatus.CREATED)

    def find_local_cookie_file(self) -> Path | None:
        for name in ("cookies.txt", "cookie.txt"):
            path = ROOT / name
            if path.exists():
                return path
        return None

    def handle_youtube_download(self) -> None:
        user = self.require_user()
        if not user:
            return

        try:
            data = self.read_request_data()
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        url = str(data.get("url", "")).strip()
        ai_base = str(data.get("ai_base", "http://localhost:8000")).strip().rstrip("/")

        if not url:
            self.send_json({"error": "YouTube URL is required."}, HTTPStatus.BAD_REQUEST)
            return

        downloads_dir = ROOT / "youtube_downloads"
        downloads_dir.mkdir(exist_ok=True)

        local_id = f"youtube_{uuid.uuid4().hex[:10]}"
        output_template = str(downloads_dir / f"{local_id}.%(ext)s")

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--merge-output-format", "mp4",
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best[ext=mp4]/best",
            "-o", output_template,
            "--print", "after_move:filepath",
            url,
        ]

        cookie_file = self.find_local_cookie_file()
        if cookie_file:
            cmd[1:1] = ["--cookies", str(cookie_file)]

        print(
            f"YouTube download started: url={url}, cookies={cookie_file or 'not found'}",
            flush=True,
        )

        try:
            process = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=900,
            )
        except FileNotFoundError:
            self.send_json(
                {"error": "yt-dlp is not installed on this Windows environment. Run: pip install yt-dlp"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        except subprocess.TimeoutExpired:
            self.send_json({"error": "YouTube download timed out."}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        output = process.stdout or ""
        if process.returncode != 0:
            tail = "\n".join(output.splitlines()[-30:])
            cookie_hint = (
                f"Using cookies: {cookie_file}"
                if cookie_file
                else "No cookies.txt/cookie.txt found beside server.py."
            )
            self.send_json(
                {"error": f"yt-dlp failed locally. {cookie_hint}\n{tail}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        downloaded_path = None
        for line in reversed(output.splitlines()):
            candidate = Path(line.strip().strip('"'))
            if candidate.exists():
                downloaded_path = candidate
                break

        if downloaded_path is None:
            candidates = sorted(
                downloads_dir.glob(f"{local_id}.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                downloaded_path = candidates[0]

        if downloaded_path is None or not downloaded_path.exists():
            self.send_json(
                {"error": "Download finished but the video file was not found."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        print(
            f"YouTube download completed: {downloaded_path} "
            f"({downloaded_path.stat().st_size / 1024 / 1024:.2f} MB)",
            flush=True,
        )

        try:
            upload_url = f"{ai_base}/upload_video"
            mime_type = mimetypes.guess_type(downloaded_path.name)[0] or "video/mp4"
            with downloaded_path.open("rb") as file_obj:
                response = requests.post(
                    upload_url,
                    files={"file": (downloaded_path.name, file_obj, mime_type)},
                    data={"title": downloaded_path.stem},
                    timeout=900,
                )
        except requests.RequestException as exc:
            self.send_json(
                {"error": f"Local download succeeded, but upload to AutoDL failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        try:
            upload_data = response.json()
        except Exception:
            upload_data = {"raw_response": response.text}

        if not response.ok:
            self.send_json(
                {"error": f"AutoDL upload failed with status {response.status_code}: {upload_data}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        print(
            f"YouTube video uploaded to AutoDL: video_id={upload_data.get('video_id')}, "
            f"filename={upload_data.get('filename')}",
            flush=True,
        )

        self.send_json({
            "status": "ready",
            "local_download_path": str(downloaded_path),
            "cookie_file_used": str(cookie_file) if cookie_file else None,
            **upload_data,
        })


    def handle_update_bookmark_log(self, path: str) -> None:
        user = self.require_user()
        if not user:
            return

        try:
            bookmark_id = int(path.rsplit("/", 1)[1])
        except ValueError:
            self.send_json({"error": "Invalid bookmark id."}, HTTPStatus.BAD_REQUEST)
            return

        data = self.read_request_data()
        if "log_text" not in data:
            self.send_json({"error": "Log text is required."}, HTTPStatus.BAD_REQUEST)
            return

        bookmark = update_bookmark_log_text(
            user_id=user["id"],
            bookmark_id=bookmark_id,
            log_text=data.get("log_text", ""),
        )
        if not bookmark:
            self.send_json({"error": "Bookmark not found."}, HTTPStatus.NOT_FOUND)
            return

        self.send_json({"bookmark": self.with_recording_url(bookmark)})

    def read_body_bytes(self, max_bytes: int = MAX_UPLOAD_BYTES * 2) -> bytes:
        """Read Content-Length or HTTP/1.1 chunked request bodies safely."""
        transfer_encoding = str(self.headers.get("Transfer-Encoding", "")).lower()

        if "chunked" in transfer_encoding:
            body = bytearray()
            while True:
                size_line = self.rfile.readline(128)
                if not size_line:
                    raise OSError("The client closed the connection during chunked upload.")

                size_token = size_line.split(b";", 1)[0].strip()
                try:
                    chunk_size = int(size_token, 16)
                except ValueError as exc:
                    raise OSError("Invalid chunked request body.") from exc

                if chunk_size == 0:
                    # Consume optional trailer headers.
                    while True:
                        trailer = self.rfile.readline(8192)
                        if trailer in {b"\r\n", b"\n", b""}:
                            break
                    break

                if len(body) + chunk_size > max_bytes:
                    raise ValueError("The upload is too large.")

                chunk = self.rfile.read(chunk_size)
                if len(chunk) != chunk_size:
                    raise OSError("The client closed the connection before a chunk completed.")
                body.extend(chunk)

                ending = self.rfile.read(2)
                if ending != b"\r\n":
                    raise OSError("Invalid chunk delimiter in request body.")

            return bytes(body)

        try:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError as exc:
            raise OSError("Invalid Content-Length header.") from exc

        if content_length <= 0:
            return b""
        if content_length > max_bytes:
            raise ValueError("The upload is too large.")

        body = bytearray()
        remaining = content_length
        while remaining > 0:
            chunk = self.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                raise OSError("The client closed the connection before the request body completed.")
            body.extend(chunk)
            remaining -= len(chunk)
        return bytes(body)

    def read_multipart_data(self) -> tuple[dict[str, str], dict | None]:
        content_type = self.headers.get("Content-Type", "")
        raw_body = self.read_body_bytes(MAX_UPLOAD_BYTES * 2)
        if not raw_body:
            raise ValueError("The upload request is empty.")
        mime_message = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8") + raw_body

        message = BytesParser(policy=policy.default).parsebytes(mime_message)
        if not message.is_multipart():
            raise ValueError("Invalid multipart upload request.")

        data: dict[str, str] = {}
        video_upload: dict | None = None

        for part in message.iter_parts():
            disposition = part.get_content_disposition()
            if disposition != "form-data":
                continue

            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""

            if filename:
                if name == "video_file":
                    video_upload = {
                        "filename": Path(filename).name,
                        "content_type": part.get_content_type(),
                        "data": payload,
                    }
            elif name:
                charset = part.get_content_charset() or "utf-8"
                try:
                    data[str(name)] = payload.decode(charset)
                except (LookupError, UnicodeDecodeError):
                    data[str(name)] = payload.decode("utf-8", errors="replace")

        return data, video_upload

    def read_request_data(self) -> dict[str, str]:
        raw_bytes = self.read_body_bytes(MAX_UPLOAD_BYTES * 2)
        content_type = self.headers.get("Content-Type", "")

        if "application/json" in content_type:
            try:
                payload = json.loads(raw_bytes.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid JSON request body.") from exc
            return {
                str(key): "" if value is None else str(value)
                for key, value in payload.items()
            }

        try:
            raw_body = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Unsupported request body encoding.") from exc

        return {
            key: values[0]
            for key, values in parse_qs(raw_body, keep_blank_values=True).items()
        }

    def save_recording(self, bookmark_id: int, upload: dict) -> Path:
        original_name = Path(str(upload.get("filename") or "camera.webm")).name
        extension = Path(original_name).suffix.lower()

        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            content_type = str(upload.get("content_type") or "").lower()
            extension = ".mp4" if "mp4" in content_type else ".webm"

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RECORDINGS_DIR / f"bookmark_{bookmark_id}{extension}"

        video_bytes = upload.get("data") or b""
        if not isinstance(video_bytes, (bytes, bytearray)):
            raise ValueError("Invalid camera recording data.")

        with destination.open("wb") as output:
            output.write(video_bytes)

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise ValueError("The uploaded camera recording is empty.")

        return destination

    def recording_path(self, bookmark_id: int) -> Path | None:
        for extension in ALLOWED_VIDEO_EXTENSIONS:
            candidate = RECORDINGS_DIR / f"bookmark_{bookmark_id}{extension}"
            if candidate.is_file():
                return candidate
        return None

    def with_recording_url(self, bookmark: dict) -> dict:
        result = dict(bookmark)
        path = self.recording_path(int(result["id"]))
        result["video_url"] = f"/recordings/{path.name}" if path else None
        return result

    def delete_recording(self, bookmark_id: int) -> None:
        path = self.recording_path(bookmark_id)
        if path:
            try:
                path.unlink()
            except OSError:
                pass
        for part_path in RECORDINGS_DIR.glob(f".bookmark_{bookmark_id}.*.part"):
            try:
                part_path.unlink()
            except OSError:
                pass

    def session_token(self) -> str | None:
        cookie_header = self.headers.get("Cookie")
        if not cookie_header:
            return None

        cookies = SimpleCookie(cookie_header)
        morsel = cookies.get(SESSION_COOKIE)
        return morsel.value if morsel else None

    def current_user(self) -> dict | None:
        return get_user_by_session(self.session_token())

    def require_user(self) -> dict | None:
        user = self.current_user()
        if not user:
            self.send_json({"error": "Login required."}, HTTPStatus.UNAUTHORIZED)
            return None
        return user

    def redirect(
        self,
        location: str,
        *,
        session_token: str | None = None,
        clear_cookie: bool = False,
    ) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        if session_token:
            self.send_header(
                "Set-Cookie",
                (
                    f"{SESSION_COOKIE}={session_token}; Path=/; HttpOnly; "
                    f"SameSite=Lax; Max-Age={SESSION_SECONDS}"
                ),
            )
        if clear_cookie:
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0",
            )
        self.end_headers()

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} - {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI Video Query local server.")
    parser.add_argument("--port", type=int, default=5500)
    args = parser.parse_args()

    db_path = init_db()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), AppHandler)
    print(f"Serving AI Video Query at http://localhost:{args.port}")
    print(f"SQLite database: {db_path}")
    print(f"Camera recordings: {RECORDINGS_DIR}")
    print("Recording upload mode: chunked-v1 (256 KB browser chunks)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()