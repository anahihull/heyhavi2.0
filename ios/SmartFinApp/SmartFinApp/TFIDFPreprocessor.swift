// TFIDFPreprocessor.swift
// Replicates the Python TfidfVectorizer (sublinear_tf=true, ngram_range=(1,3),
// strip_accents='unicode') so IntentClassifier and SentimentClassifier can run
// fully on-device without a server round-trip.
//
// Usage:
//   let vector = try TFIDFPreprocessor.vectorize(
//                        text: "Quiero cancelar mi cuenta",
//                        modelURL: IntentClassifier.urlOfModelInThisBundle)
//   let result = try IntentClassifier().prediction(tfidf_input: vector)
//   print(result.intent)

import Foundation
import CoreML

// MARK: - Cache (load vocab/IDF once per model URL)
private var cache: [URL: TFIDFCache] = [:]
private struct TFIDFCache {
    let vocab: [String: Int]   // token → feature index
    let idf: [Float]           // index → idf weight
    let ngramMin: Int
    let ngramMax: Int
    let nFeatures: Int
}

// MARK: - Public API
enum TFIDFPreprocessor {

    /// Tokenize `text`, compute TF-IDF, and return an MLMultiArray
    /// sized to the model's feature dimension.
    ///
    /// - Parameters:
    ///   - text: Raw customer message in Spanish.
    ///   - modelURL: URL to the .mlmodelc compiled folder or the .mlmodel
    ///               file whose `userInfo` contains vocabulary metadata.
    static func vectorize(text: String, modelURL: URL) throws -> MLMultiArray {
        let meta = try loadMeta(from: modelURL)
        let tokens = tokenize(text)
        let ngrams = generateNgrams(tokens, min: meta.ngramMin, max: meta.ngramMax)

        // Count term frequencies
        var tf = [Int](repeating: 0, count: meta.nFeatures)
        for gram in ngrams {
            if let idx = meta.vocab[gram] {
                tf[idx] += 1
            }
        }

        // Apply sublinear TF  →  1 + log(count)  (matches sklearn's sublinear_tf=True)
        // then multiply by IDF weight, then L2-normalize
        var raw = [Float](repeating: 0, count: meta.nFeatures)
        for i in 0 ..< meta.nFeatures where tf[i] > 0 {
            raw[i] = (1.0 + log(Float(tf[i]))) * meta.idf[i]
        }

        // L2 normalize
        let norm = sqrt(raw.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            for i in 0 ..< meta.nFeatures { raw[i] /= norm }
        }

        let array = try MLMultiArray(shape: [NSNumber(value: meta.nFeatures)],
                                     dataType: .float32)
        for i in 0 ..< meta.nFeatures {
            array[i] = NSNumber(value: raw[i])
        }
        return array
    }
}

// MARK: - Private helpers

private func loadMeta(from modelURL: URL) throws -> TFIDFCache {
    if let hit = cache[modelURL] { return hit }

    // Locate the CoreML model's userInfo (metadata) stored in coremldata.bin
    // CoreML exposes it via MLModel.modelDescription.metadata after compiling,
    // but the easiest path is to load the uncompiled .mlmodel's metadata JSON
    // that we store alongside the model in the app bundle.
    //
    // Convention: we bundle a sidecar file "<ModelName>_meta.json" next to the model.
    // export_coreml.py writes the vocabulary into user_defined_metadata, which
    // is accessible via MLModel after compilation.

    let compiledURL: URL
    if modelURL.pathExtension == "mlmodel" {
        compiledURL = try MLModel.compileModel(at: modelURL)
    } else {
        compiledURL = modelURL   // already compiled (.mlmodelc)
    }

    let mlmodel = try MLModel(contentsOf: compiledURL)
    let meta = mlmodel.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String] ?? [:]

    guard
        let vocabJSON  = meta["vocabulary_json"],
        let idfJSON    = meta["idf_weights_json"],
        let nFeatStr   = meta["n_features"],
        let nFeatures  = Int(nFeatStr),
        let ngramMinStr = meta["ngram_range_min"],
        let ngramMaxStr = meta["ngram_range_max"],
        let ngramMin   = Int(ngramMinStr),
        let ngramMax   = Int(ngramMaxStr)
    else {
        throw TFIDFError.missingMetadata(modelURL.lastPathComponent)
    }

    let vocabData = Data(vocabJSON.utf8)
    let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)

    let idfData = Data(idfJSON.utf8)
    let idf = try JSONDecoder().decode([Float].self, from: idfData)

    let result = TFIDFCache(vocab: vocab, idf: idf,
                             ngramMin: ngramMin, ngramMax: ngramMax,
                             nFeatures: nFeatures)
    cache[modelURL] = result
    return result
}

/// Lowercase, strip accents (NFD + remove Mn), then split into word tokens ≥ 2 chars.
/// Matches sklearn's default tokenizer with strip_accents='unicode'.
private func tokenize(_ text: String) -> [String] {
    let lowered = text.lowercased()
    // NFD decompose → remove combining marks (accents)
    let stripped = lowered
        .decomposedStringWithCompatibilityMapping
        .unicodeScalars
        .filter { !CharacterSet.nonBaseCharacters.contains($0) }
    let cleaned = String(String.UnicodeScalarView(stripped))

    // Split on non-alphanumeric, keep tokens with ≥ 2 characters
    let parts = cleaned.components(separatedBy: CharacterSet.alphanumerics.inverted)
    return parts.filter { $0.count >= 2 }
}

/// Generate n-grams from `tokens` for n in [min, max].
private func generateNgrams(_ tokens: [String], min: Int, max: Int) -> [String] {
    var ngrams: [String] = []
    for n in min ... max {
        guard tokens.count >= n else { continue }
        for i in 0 ... (tokens.count - n) {
            ngrams.append(tokens[i ..< i + n].joined(separator: " "))
        }
    }
    return ngrams
}

// MARK: - Error
enum TFIDFError: LocalizedError {
    case missingMetadata(String)
    var errorDescription: String? {
        switch self {
        case .missingMetadata(let name):
            return "TFIDFPreprocessor: vocabulary metadata not found in \(name). Re-run export_coreml.py."
        }
    }
}
