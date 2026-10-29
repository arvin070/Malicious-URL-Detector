import Foundation
import URLGuardCore

/// Talks to the same Flask backend the web UI and the Android app use.
///
/// There is no API key in this file and none in the app bundle. The only
/// credential is the user's own access token, obtained by signing in and kept
/// in the Keychain -- the iOS counterpart of the Android `SessionStore`.
/// Embedding a shared secret in a client that ships to devices would mean
/// publishing it.
public actor ApiClient {

    public struct ApiError: Error, LocalizedError {
        public let status: Int
        public let code: String?
        public let message: String
        public var errorDescription: String? { message }
    }

    public static let shared = ApiClient()

    private var baseURL: URL?
    private let session: URLSession
    // The access group must match `keychain-access-groups` in the
    // entitlements, or the Share and Safari extensions get their own private
    // Keychain and the user appears signed out inside them.
    private let keychain = Keychain(service: "com.aics.urlguard",
                                    accessGroup: Keychain.sharedAccessGroup)

    private init() {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 15
        cfg.timeoutIntervalForResource = 30
        // No cookies: the API is token-authenticated, and a cookie jar would
        // be one more place a credential could end up.
        cfg.httpCookieAcceptPolicy = .never
        cfg.httpShouldSetCookies = false
        self.session = URLSession(configuration: cfg)
    }

    public func configure(baseURL: URL) { self.baseURL = baseURL }

    private func request(_ method: String, _ path: String,
                         body: [String: Any]? = nil,
                         authenticated: Bool = true) async throws -> [String: Any] {
        guard let base = baseURL else {
            throw ApiError(status: 0, code: "no_server",
                           message: "No server address has been configured.")
        }
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if authenticated, let token = keychain.get("access") {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }

        let (data, response) = try await session.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let json = (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any] ?? [:]

        if status == 401, authenticated, try await refresh() {
            return try await request(method, path, body: body,
                                     authenticated: authenticated)
        }
        guard (200..<300).contains(status) else {
            throw ApiError(status: status,
                           code: json["code"] as? String,
                           message: json["error"] as? String
                                ?? "The server returned \(status).")
        }
        return json
    }

    @discardableResult
    private func refresh() async throws -> Bool {
        guard let token = keychain.get("refresh") else { return false }
        guard let base = baseURL else { return false }
        var req = URLRequest(url: base.appendingPathComponent("/api/auth/refresh"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(
            withJSONObject: ["refresh_token": token])
        let (data, response) = try await session.data(for: req)
        guard (response as? HTTPURLResponse)?.statusCode == 200,
              let json = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any] else {
            // Refresh failed: clear the stale pair rather than retrying with
            // a token the server has rotated away.
            keychain.delete("access"); keychain.delete("refresh")
            return false
        }
        store(json)
        return true
    }

    private func store(_ json: [String: Any]) {
        if let a = json["access_token"] as? String { keychain.set("access", a) }
        if let r = json["refresh_token"] as? String { keychain.set("refresh", r) }
    }

    // MARK: - endpoints (mirror the Android ApiClient)

    public func login(email: String, password: String) async throws -> [String: Any] {
        let json = try await request("POST", "/api/auth/login",
                                     body: ["email": email, "password": password],
                                     authenticated: false)
        store(json)
        return json
    }

    public func register(name: String, email: String,
                         password: String) async throws -> [String: Any] {
        let json = try await request("POST", "/api/auth/register",
                                     body: ["name": name, "email": email,
                                            "password": password],
                                     authenticated: false)
        store(json)
        return json
    }

    public func logout() async {
        _ = try? await request("POST", "/api/auth/logout")
        keychain.delete("access"); keychain.delete("refresh")
    }

    public var isSignedIn: Bool { keychain.get("access") != nil }

    /// `analyzeWebsite` asks the BACKEND to retrieve and analyse the page. The
    /// app never downloads or renders an arbitrary page itself: doing that on
    /// the device would pull attacker-controlled content into the process
    /// meant to be protecting the user. The backend fetch is GET-only, runs no
    /// JavaScript and is SSRF-guarded.
    public func scan(url: String, scope: String = "url",
                     analyzeWebsite: Bool = false) async throws -> [String: Any] {
        try await request("POST", "/api/scan",
                          body: ["url": url, "scope": scope,
                                 "analyze_website": analyzeWebsite])
    }

    public func history(limit: Int = 25, offset: Int = 0) async throws -> [Any] {
        guard let base = baseURL else { return [] }
        var comps = URLComponents(
            url: base.appendingPathComponent("/api/history"),
            resolvingAgainstBaseURL: false)!
        comps.queryItems = [
            .init(name: "limit", value: String(limit)),
            .init(name: "offset", value: String(offset)),
        ]
        var req = URLRequest(url: comps.url!)
        if let token = keychain.get("access") {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, _) = try await session.data(for: req)
        return (try? JSONSerialization.jsonObject(with: data)) as? [Any] ?? []
    }

    public func meta() async throws -> [String: Any] {
        try await request("GET", "/api/meta", authenticated: false)
    }
}
