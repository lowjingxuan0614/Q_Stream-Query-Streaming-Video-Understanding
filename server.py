from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
)


ROOT = Path(__file__).resolve().parent
SESSION_COOKIE = "ai_video_session"


class AppHandler(SimpleHTTPRequestHandler):
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
            self.send_json({"bookmarks": list_bookmarks(user["id"])})
            return

        if path in {"/main.html", "/savedLog.html"} and not self.current_user():
            self.redirect("/login.html")
            return

        super().do_GET()

    def do_POST(self) -> None:
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

        data = self.read_request_data()

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

        self.send_json({"bookmark": bookmark}, HTTPStatus.CREATED)

    def read_request_data(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw_body = self.rfile.read(length).decode("utf-8") if length else ""
        content_type = self.headers.get("Content-Type", "")

        if "application/json" in content_type:
            return json.loads(raw_body or "{}")

        return {
            key: values[0]
            for key, values in parse_qs(raw_body, keep_blank_values=True).items()
        }

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
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI Video Query local server.")
    parser.add_argument("--port", type=int, default=5500)
    args = parser.parse_args()

    db_path = init_db()
    server = ThreadingHTTPServer(("localhost", args.port), AppHandler)
    print(f"Serving AI Video Query at http://localhost:{args.port}")
    print(f"SQLite database: {db_path}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
