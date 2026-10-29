package com.aics.urlguard

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/**
 * On-device URL classifier: 122 lexical features -> ONNX tree ensemble -> score.
 *
 * Nothing here contacts the network. The URL is never resolved, fetched or
 * opened; it is treated as a string, exactly as in the Python pipeline. That is
 * what makes it safe to classify a link BEFORE the user visits it.
 */
class UrlClassifier(
    private val session: OrtSession,
    private val env: OrtEnvironment,
    private val tld: TldExtract,
    val featureNames: List<String>,
    val threshold: Double,
    val modelName: String,
) : AutoCloseable {

    data class Verdict(
        val url: String,
        val score: Double,
        val malicious: Boolean,
        val threshold: Double,
        /** Highest-contributing feature values, for the "why" panel. */
        val topFeatures: List<Pair<String, Double>>,
        /** Every measured feature, so signal text can be built from real values. */
        val features: Map<String, Double>,
    )

    /**
     * Features are looked up BY NAME and written in the order the exported
     * model declares, so a reordering in either language can never silently
     * feed the model a shuffled vector.
     */
    fun featureVector(url: String): FloatArray {
        val f = UrlFeatures.extract(url, tld)
        val out = FloatArray(featureNames.size)
        for (i in featureNames.indices) {
            val v = f[featureNames[i]]
                ?: error("feature '${featureNames[i]}' missing from the extractor")
            out[i] = v.toFloat()
        }
        return out
    }

    fun classify(url: String): Verdict {
        val vec = featureVector(url)
        val score = score(vec)
        val named = featureNames.zip(vec.map { it.toDouble() })
        val all = named.toMap()
        return Verdict(
            url = url,
            score = score,
            malicious = score >= threshold,
            threshold = threshold,
            topFeatures = INTERESTING.mapNotNull { n ->
                named.firstOrNull { it.first == n }
            },
            features = all,
        )
    }

    @Synchronized
    fun score(vec: FloatArray): Double {
        val shape = longArrayOf(1, vec.size.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(vec), shape).use { tensor ->
            session.run(mapOf(session.inputNames.first() to tensor)).use { res ->
                return positiveProbability(res)
            }
        }
    }

    /**
     * Reads the positive-class probability.
     *
     * The converter may emit probabilities either as a plain [N,2] tensor or,
     * when ZipMap is enabled, as a sequence of {label -> probability} maps.
     * Both shapes are handled: which one you get depends on the converter
     * version, and an app that only understood one would break on a model
     * re-export rather than at build time.
     */
    private fun positiveProbability(res: OrtSession.Result): Double {
        val value: OnnxValue = res.firstOrNull { it.key == "probabilities" }?.value
            ?: res.firstOrNull { it.key != "label" }?.value
            ?: error("ONNX model produced no probability output")

        when (val v = value.value) {
            is Array<*> -> {                       // float[1][2]
                val row = v.firstOrNull() as? FloatArray
                    ?: error("unexpected probability tensor shape")
                return row[if (row.size > 1) 1 else 0].toDouble()
            }
            is List<*> -> {                        // ZipMap: List<Map<Long, Float>>
                val m = v.firstOrNull() as? Map<*, *>
                    ?: error("unexpected ZipMap output")
                val entry = m.entries.firstOrNull { (k, _) ->
                    (k as? Number)?.toInt() == 1 || k.toString() == "1"
                } ?: m.entries.last()
                return (entry.value as Number).toDouble()
            }
            is FloatArray -> return v[if (v.size > 1) 1 else 0].toDouble()
            else -> error("unsupported probability output: ${v?.javaClass}")
        }
    }

    override fun close() {
        runCatching { session.close() }
    }

    companion object {
        /** Shown in the "why was this flagged" panel -- human-readable signals. */
        private val INTERESTING = listOf(
            "suspicious_word_count", "brand_outside_domain", "has_ip",
            "has_at_symbol", "num_hyphens", "subdomain_depth",
            "has_suspicious_port", "is_punycode", "has_confusable_chars",
            "url_length",
        )

        fun create(
            modelBytes: ByteArray,
            suffixListLines: Sequence<String>,
            featureNames: List<String>,
            threshold: Double,
            modelName: String,
        ): UrlClassifier {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                // One thread: scoring a single URL is microseconds of work, and
                // spinning up a thread pool per navigation would cost far more
                // than it saves. It also keeps scores bit-reproducible, the same
                // property train.py enforces with n_jobs=1.
                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)
            }
            val session = env.createSession(modelBytes, opts)
            val tld = TldExtract.fromSuffixLines(suffixListLines)
            return UrlClassifier(session, env, tld, featureNames, threshold, modelName)
        }
    }
}
