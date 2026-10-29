import Foundation

/// Local scan history, shared with the extensions through an App Group.
///
/// Stored on the device only. The server keeps its own history for a signed-in
/// account; this is what the app can show without an account and without a
/// network call.
final class ScanHistory {
    static let shared = ScanHistory(suiteName: "group.com.aics.urlguard")

    struct Entry: Codable, Identifiable {
        let id: UUID
        let url: String
        let score: Int
        let band: String
        let at: Date
    }

    static let max = 100
    private let defaults: UserDefaults
    private let key = "scan_history"

    init(suiteName: String) {
        self.defaults = UserDefaults(suiteName: suiteName) ?? .standard
    }

    func all() -> [Entry] {
        guard let data = defaults.data(forKey: key),
              let items = try? JSONDecoder().decode([Entry].self, from: data)
        else { return [] }
        return items
    }

    func add(url: String, score: Int, band: String) {
        var items = all()
        items.insert(Entry(id: UUID(), url: url, score: score, band: band,
                           at: Date()), at: 0)
        if items.count > Self.max { items = Array(items.prefix(Self.max)) }
        if let data = try? JSONEncoder().encode(items) {
            defaults.set(data, forKey: key)
        }
    }

    func clear() { defaults.removeObject(forKey: key) }
}
