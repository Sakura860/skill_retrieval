"""Deterministic loopback HTTP service used by isolated benchmark tasks."""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class LocalHTTPFixture:
    """Serve an exact route table on a random loopback port and record requests."""

    def __init__(self, fixture: dict[str, Any]):
        if fixture.get("type") != "local_http_api":
            raise ValueError("HTTP fixture.type 必须是 local_http_api")
        routes = fixture.get("routes")
        if not isinstance(routes, list) or not routes:
            raise ValueError("HTTP fixture.routes 必须是非空列表")
        self.routes = [self._normalize_route(route) for route in routes]
        keys = [(route["method"], route["path"]) for route in self.routes]
        if len(keys) != len(set(keys)):
            raise ValueError("HTTP fixture 路由 method/path 不能重复")
        self.requests: list[dict[str, Any]] = []
        self.route_call_counts = {
            f"{route['method']} {route['path']}": 0 for route in self.routes
        }
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _normalize_route(route: Any) -> dict[str, Any]:
        if not isinstance(route, dict):
            raise TypeError("HTTP route 必须是对象")
        method = str(route.get("method", "GET")).upper()
        path = str(route.get("path", ""))
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
            raise ValueError(f"不支持的 HTTP method: {method}")
        if not path.startswith("/") or "://" in path:
            raise ValueError("HTTP fixture path 必须是本地绝对路径")
        status = route.get("status", 200)
        if not isinstance(status, int) or not 100 <= status <= 599:
            raise ValueError("HTTP route.status 必须是 100-599")
        normalized = dict(route)
        normalized.update({"method": method, "path": path, "status": status})
        required_headers = normalized.get("required_headers", {})
        if not isinstance(required_headers, dict):
            raise TypeError("required_headers 必须是对象")
        normalized["required_headers"] = {
            str(key).lower(): str(value) for key, value in required_headers.items()
        }
        return normalized

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("HTTP fixture 尚未启动")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                fixture._handle(self)

            def do_POST(self) -> None:  # noqa: N802
                fixture._handle(self)

            def do_PUT(self) -> None:  # noqa: N802
                fixture._handle(self)

            def do_PATCH(self) -> None:  # noqa: N802
                fixture._handle(self)

            def do_DELETE(self) -> None:  # noqa: N802
                fixture._handle(self)

            def do_HEAD(self) -> None:  # noqa: N802
                fixture._handle(self)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="skill-agent-http-fixture",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": deepcopy(self.requests),
                "route_call_counts": dict(self.route_call_counts),
            }

    def _handle(self, request: BaseHTTPRequestHandler) -> None:
        path = request.path
        method = request.command.upper()
        route = next(
            (
                item for item in self.routes
                if item["method"] == method and item["path"] == path
            ),
            None,
        )
        body = self._read_body(request)
        relevant_headers = {
            name: request.headers.get(name)
            for name in ("authorization", "idempotency-key")
            if request.headers.get(name) is not None
        }
        status = 404
        response_json: Any = {"error": "route_not_found"}
        response_text: str | None = None
        if route is not None:
            status, response_json, response_text = self._route_response(
                route,
                relevant_headers,
                body,
            )
        record = {
            "method": method,
            "path": path,
            "headers": relevant_headers,
            "json": body,
            "response_status": status,
        }
        with self._lock:
            self.requests.append(record)
            if route is not None:
                key = f"{method} {path}"
                self.route_call_counts[key] += 1

        payload = (
            response_text.encode("utf-8")
            if response_text is not None
            else json.dumps(response_json, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        )
        request.send_response(status)
        request.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
            if response_text is not None
            else "application/json; charset=utf-8",
        )
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        if method != "HEAD":
            request.wfile.write(payload)

    @staticmethod
    def _read_body(request: BaseHTTPRequestHandler) -> Any:
        length = int(request.headers.get("Content-Length", "0") or 0)
        if length == 0:
            return None
        raw = request.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    @staticmethod
    def _route_response(
        route: dict[str, Any],
        headers: dict[str, str],
        body: Any,
    ) -> tuple[int, Any, str | None]:
        for name, expected in route["required_headers"].items():
            if headers.get(name) != expected:
                return 401, {"error": "invalid_header", "header": name}, None
        if "expected_json" in route and body != route["expected_json"]:
            return 400, {"error": "invalid_json_body"}, None
        return (
            route["status"],
            deepcopy(route.get("response_json")),
            route.get("response_text"),
        )
