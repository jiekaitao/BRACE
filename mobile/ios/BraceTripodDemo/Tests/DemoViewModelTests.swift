import XCTest
@testable import BraceTripodDemo

@MainActor
final class DemoViewModelTests: XCTestCase {
    func testValidatedUploadBaseURLAcceptsHTTPS() throws {
        let url = try DemoViewModel.validatedUploadBaseURL(from: "https://demo.example.com:8443")
        XCTAssertEqual(url.scheme, "https")
        XCTAssertEqual(url.host, "demo.example.com")
    }

    func testValidatedUploadBaseURLRejectsNonHTTP() {
        XCTAssertThrowsError(try DemoViewModel.validatedUploadBaseURL(from: "ftp://demo.example.com"))
    }

    func testDefaultLiveStreamURLConvertsScheme() throws {
        let uploadURL = try DemoViewModel.validatedUploadBaseURL(from: "https://demo.example.com:8443")
        let liveURL = try DemoViewModel.defaultLiveStreamURL(fromUploadBaseURL: uploadURL)
        XCTAssertEqual(liveURL.absoluteString, "wss://demo.example.com:8443/live-stream")
    }
}
