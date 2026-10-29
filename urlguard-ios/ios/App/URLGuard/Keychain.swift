import Foundation
import Security

/// Minimal Keychain wrapper for the auth tokens.
///
/// Tokens go in the Keychain rather than UserDefaults because UserDefaults is
/// a plist in the app container: readable from a backup and from any process
/// that gets at the container. `kSecAttrAccessibleAfterFirstUnlock` lets the
/// Share and Safari extensions read the token when they run in the background,
/// while still keeping it unreadable before the device has been unlocked once
/// since boot.
///
/// The access group is what lets the extensions share the signed-in session
/// with the main app; it must match the one in the entitlements files.
struct Keychain {
    let service: String
    var accessGroup: String? = nil

    /// Resolved at runtime from the bundle's own team prefix, because the
    /// entitlement value `$(AppIdentifierPrefix)com.aics.urlguard` is expanded
    /// at build time and cannot be hard-coded here without pinning a team id.
    static var sharedAccessGroup: String? {
        guard let prefix = Bundle.main
            .object(forInfoDictionaryKey: "AppIdentifierPrefix") as? String
        else { return nil }
        return "\(prefix)com.aics.urlguard"
    }

    private func query(_ key: String) -> [String: Any] {
        var q: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        if let accessGroup { q[kSecAttrAccessGroup as String] = accessGroup }
        return q
    }

    func get(_ key: String) -> String? {
        var q = query(key)
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess,
              let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func set(_ key: String, _ value: String) {
        let q = query(key)
        SecItemDelete(q as CFDictionary)
        var add = q
        add[kSecValueData as String] = Data(value.utf8)
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(add as CFDictionary, nil)
    }

    func delete(_ key: String) {
        SecItemDelete(query(key) as CFDictionary)
    }
}
