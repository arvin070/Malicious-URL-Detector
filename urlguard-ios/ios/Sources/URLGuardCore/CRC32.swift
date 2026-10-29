import Foundation

/// CRC-32/ISO-HDLC, the same checksum Python's `zlib.crc32` computes.
///
/// The hashed n-gram features bucket a string by `zlib.crc32(s) % buckets`.
/// Python's own `hash()` is salted per process and would differ between runs,
/// which is why neither the Python side nor any port uses it. Java supplies
/// `java.util.zip.CRC32` so the Kotlin port got this for free; Foundation does
/// not, so it is implemented here.
///
/// Verified against the Python extractor by the golden-vector test: any
/// divergence moves every `cng3_*` and `tng2_*` feature at once, which is
/// loud rather than subtle.
enum CRC32 {

    private static let table: [UInt32] = {
        var t = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1) == 1 ? (0xEDB8_8320 ^ (c >> 1)) : (c >> 1)
            }
            t[i] = c
        }
        return t
    }()

    /// Checksum of the UTF-8 bytes of `s`, matching `zlib.crc32(s.encode())`.
    static func checksum(_ s: String) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in Array(s.utf8) {
            let idx = Int((crc ^ UInt32(byte)) & 0xFF)
            crc = table[idx] ^ (crc >> 8)
        }
        return crc ^ 0xFFFF_FFFF
    }

    /// `zlib.crc32(s) % mod` -- the bucket index used by the n-gram features.
    static func bucket(_ s: String, mod: Int) -> Int {
        Int(checksum(s) % UInt32(mod))
    }
}
