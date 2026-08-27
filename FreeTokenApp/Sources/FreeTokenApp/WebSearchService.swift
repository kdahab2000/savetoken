import Foundation

struct WebSearchResult: Decodable {
    let AbstractText: String?
    let AbstractURL: String?
    let Heading: String?
    let RelatedTopics: [RelatedTopic]?

    struct RelatedTopic: Decodable {
        let Text: String?
        let FirstURL: String?
        let Topics: [RelatedTopic]?
    }
}

enum WebSearchService {
    static func search(_ query: String) async throws -> String {
        var components = URLComponents(string: "https://api.duckduckgo.com/")!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "format", value: "json"),
            URLQueryItem(name: "no_html", value: "1"),
            URLQueryItem(name: "skip_disambig", value: "1")
        ]
        var request = URLRequest(url: components.url!)
        request.timeoutInterval = 15
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        let result = try JSONDecoder().decode(WebSearchResult.self, from: data)
        var lines = ["Web results for: \(query)"]
        if let heading = result.Heading, let abstract = result.AbstractText, !abstract.isEmpty {
            lines.append("\(heading): \(abstract)")
            if let url = result.AbstractURL { lines.append("Source: \(url)") }
        }
        func append(_ topics: [WebSearchResult.RelatedTopic]?) {
            for topic in topics ?? [] {
                if let nested = topic.Topics { append(nested) }
                else if let text = topic.Text, !text.isEmpty {
                    lines.append("- \(text)\(topic.FirstURL.map { " (\($0))" } ?? "")")
                }
                if lines.count >= 9 { return }
            }
        }
        append(result.RelatedTopics)
        return lines.count == 1 ? lines[0] + "\nNo concise result was returned." : lines.joined(separator: "\n")
    }
}
