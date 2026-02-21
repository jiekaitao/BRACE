import Foundation

struct PendingClipUpload: Codable, Identifiable {
    let id: String
    let filePath: String
    let playID: String
    let playerID: String
    let coachEmail: String?
    var attempts: Int
    var nextAttemptAt: Date
    let createdAt: Date
}

actor TripodUploadQueue {
    private let queueFileURL: URL
    private let clipsDirectoryURL: URL
    private var items: [PendingClipUpload] = []

    init() {
        let fm = FileManager.default
        let baseDir: URL
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            baseDir = appSupport.appendingPathComponent("BRACETripodQueue", isDirectory: true)
        } else {
            baseDir = fm.temporaryDirectory.appendingPathComponent("BRACETripodQueue", isDirectory: true)
        }
        clipsDirectoryURL = baseDir.appendingPathComponent("clips", isDirectory: true)
        queueFileURL = baseDir.appendingPathComponent("queue.json")
        try? fm.createDirectory(at: clipsDirectoryURL, withIntermediateDirectories: true)
        items = Self.loadQueue(from: queueFileURL)
    }

    func enqueue(clipURL: URL, playID: String, playerID: String, coachEmail: String?) -> PendingClipUpload? {
        let id = UUID().uuidString
        let destination = clipsDirectoryURL.appendingPathComponent("\(id).mp4")
        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: clipURL, to: destination)
            let item = PendingClipUpload(
                id: id,
                filePath: destination.path,
                playID: playID,
                playerID: playerID,
                coachEmail: coachEmail,
                attempts: 0,
                nextAttemptAt: Date(),
                createdAt: Date()
            )
            items.append(item)
            persistQueue()
            return item
        } catch {
            return nil
        }
    }

    func dueItems(now: Date = Date()) -> [PendingClipUpload] {
        items
            .filter { $0.nextAttemptAt <= now }
            .sorted(by: { $0.createdAt < $1.createdAt })
    }

    func markSuccess(id: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        let item = items.remove(at: index)
        try? FileManager.default.removeItem(atPath: item.filePath)
        persistQueue()
    }

    func markFailure(id: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        item.attempts += 1
        let backoffSeconds = min(pow(2.0, Double(item.attempts)) * 5.0, 300.0)
        item.nextAttemptAt = Date().addingTimeInterval(backoffSeconds)
        items[index] = item
        persistQueue()
    }

    func hasPending() -> Bool {
        !items.isEmpty
    }

    func pendingCount() -> Int {
        items.count
    }

    private static func loadQueue(from queueFileURL: URL) -> [PendingClipUpload] {
        guard let data = try? Data(contentsOf: queueFileURL) else { return [] }
        let decoder = JSONDecoder()
        return (try? decoder.decode([PendingClipUpload].self, from: data)) ?? []
    }

    private func persistQueue() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        guard let data = try? encoder.encode(items) else { return }
        try? data.write(to: queueFileURL, options: .atomic)
    }
}
