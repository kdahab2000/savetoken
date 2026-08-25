import Foundation

struct SSHHost: Identifiable, Equatable, Sendable {
    let alias: String

    var id: String { alias }
}

struct SSHCommandResult: Equatable, Sendable {
    let alias: String
    let command: String
    let output: String
    let exitCode: Int32

    var succeeded: Bool { exitCode == 0 }
}

struct SSHConfirmation: Identifiable, Equatable {
    enum Source: String {
        case manual = "You"
        case model = "The local Ollama model"
    }

    let alias: String
    let command: String
    let source: Source

    var id: String { "\(source.rawValue)-\(alias)-\(command)" }
}

enum SSHClientError: Error, LocalizedError, Equatable, Sendable {
    case noSSHConfig
    case unknownAlias(String)
    case emptyCommand
    case couldNotStart(String)

    var errorDescription: String? {
        switch self {
        case .noSSHConfig:
            return "No SSH configuration was found at ~/.ssh/config."
        case .unknownAlias(let alias):
            return "SSH alias \(alias) is not present in ~/.ssh/config."
        case .emptyCommand:
            return "The SSH command is empty."
        case .couldNotStart(let message):
            return "Could not start ssh: \(message)"
        }
    }
}

/// Executes only through the user's existing OpenSSH configuration. The app
/// never handles passwords or private-key contents, and BatchMode prevents an
/// invisible password prompt from blocking the UI.
struct SSHClient: Sendable {
    static let sshExecutable = "/usr/bin/ssh"
    static let identityCommand = "hostname; id -un; pwd"
    static let commandTimeout: TimeInterval = 45

    private let configURL: URL

    init(homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) {
        self.configURL = homeDirectory
            .appendingPathComponent(".ssh", isDirectory: true)
            .appendingPathComponent("config")
    }

    func configuredAliases() -> [SSHHost] {
        guard let text = try? String(contentsOf: configURL, encoding: .utf8) else {
            return []
        }
        return Self.parseAliases(text).map(SSHHost.init(alias:))
    }

    static func parseAliases(_ contents: String) -> [String] {
        var aliases: [String] = []
        var seen = Set<String>()

        for rawLine in contents.split(whereSeparator: \.isNewline) {
            let line = rawLine.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false)[0]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let fields = line.split { $0 == " " || $0 == "\t" }
            guard let directive = fields.first?.lowercased(), directive == "host" else {
                continue
            }
            for field in fields.dropFirst() {
                let alias = String(field)
                guard !alias.isEmpty,
                      !alias.hasPrefix("!"),
                      !alias.contains("*"),
                      !alias.contains("?") else { continue }
                if seen.insert(alias).inserted {
                    aliases.append(alias)
                }
            }
        }
        return aliases
    }

    func runIdentityCheck(alias: String) async throws -> SSHCommandResult {
        try await run(alias: alias, command: Self.identityCommand)
    }

    func run(alias: String, command: String) async throws -> SSHCommandResult {
        let aliases = configuredAliases().map(\.alias)
        guard aliases.contains(alias) else {
            throw SSHClientError.unknownAlias(alias)
        }
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw SSHClientError.emptyCommand }

        return try await Task.detached(priority: .userInitiated) {
            try Self.runSynchronously(alias: alias, command: trimmed)
        }.value
    }

    private static func runSynchronously(alias: String, command: String) throws -> SSHCommandResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: sshExecutable)
        process.arguments = [
            "-T",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=2",
            alias,
            command
        ]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        let timeout = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timeout.schedule(deadline: .now() + commandTimeout)
        timeout.setEventHandler {
            if process.isRunning { process.terminate() }
        }

        do {
            try process.run()
        } catch {
            timeout.cancel()
            throw SSHClientError.couldNotStart(error.localizedDescription)
        }
        timeout.resume()

        // Reading while the process runs prevents a verbose remote command
        // from filling the pipe and deadlocking ssh.
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        timeout.cancel()

        let output = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return SSHCommandResult(
            alias: alias,
            command: command,
            output: output.isEmpty ? "(no output)" : output,
            exitCode: process.terminationStatus)
    }
}
