import SwiftUI

struct HistoryView: View {
    @State private var items = ScanHistory.shared.all()

    var body: some View {
        NavigationStack {
            Group {
                if items.isEmpty {
                    ContentUnavailableView(
                        "No scans yet",
                        systemImage: "clock",
                        description: Text("Scans you run on this device appear "
                                          + "here. Nothing is counted until "
                                          + "you run one."))
                } else {
                    List(items) { item in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.url).font(.callout).lineLimit(2)
                            HStack {
                                Text(item.band.uppercased())
                                    .font(.caption2.bold())
                                    .foregroundStyle(colour(item.band))
                                Text("\(item.score)/100").font(.caption2)
                                    .monospacedDigit()
                                Spacer()
                                Text(item.at, style: .relative)
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("History")
            .toolbar {
                if !items.isEmpty {
                    Button("Clear") { ScanHistory.shared.clear(); items = [] }
                }
            }
            .onAppear { items = ScanHistory.shared.all() }
        }
    }

    private func colour(_ band: String) -> Color {
        switch band {
        case "low": return .green
        case "suspicious": return .orange
        case "high": return .red
        default: return .gray
        }
    }
}
