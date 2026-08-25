import XCTest
@testable import FreeTokenApp

final class SSHTests: XCTestCase {
    func testSSHConfigParserKeepsNamedAliasesAndIgnoresPatterns() {
        let config = """
        Host *
            ServerAliveInterval 60
        Host hostinger-vps gpu dsw-cpu
            User khaled
        Host !blocked *.example.com ?wildcard
        Host hostinger-vps
        """

        XCTAssertEqual(
            SSHClient.parseAliases(config),
            ["hostinger-vps", "gpu", "dsw-cpu"])
    }

    func testSSHIdentityCommandIsReadOnlyIdentityProbe() {
        XCTAssertEqual(SSHClient.identityCommand, "hostname; id -un; pwd")
    }

    func testSSHResultSuccessUsesZeroExitCode() {
        let result = SSHCommandResult(
            alias: "gpu", command: "hostname", output: "example", exitCode: 0)
        XCTAssertTrue(result.succeeded)
    }
}
