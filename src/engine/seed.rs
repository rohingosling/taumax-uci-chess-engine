//----------------------------------------------------------------------------------------------------------------------
// Project: Taumax
// Version: 1.0.0
// Date:    2024
// Author:  Rohin Gosling
//
// Description:
//
//   Deterministic seed handling for reproducible Taumax experiments.
//
//----------------------------------------------------------------------------------------------------------------------

use rand::rngs::StdRng;
use rand::SeedableRng;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

//----------------------------------------------------------------------------------------------------------------------
// Function: deterministic_seed_value
//
// Description:
//
//   Convert seed text into a deterministic 64-bit value.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn deterministic_seed_value(seed_text: &str) -> u64 {

    // Compute one stable numeric seed by hashing the trimmed text bytes with FNV-1a.

    fnv1a_64(seed_text.trim().as_bytes())
}

//----------------------------------------------------------------------------------------------------------------------
// Function: deterministic_seed_bytes
//
// Description:
//
//   Expand seed text and a stream name into the 32 bytes required by StdRng.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn deterministic_seed_bytes(seed_text: &str, stream_name: &str) -> [u8; 32] {

    // Normalize both text inputs so incidental whitespace does not create a different random stream.

    let normalized_seed_text = seed_text.trim();
    let normalized_stream_name = stream_name.trim();
    let mut seed_bytes = [0_u8; 32];

    for chunk_index in 0..4 {

        // Compute one eight-byte chunk by hashing seed, stream, and chunk index together. Four chunks
        // make the 32 bytes required by StdRng.

        let chunk_text = format!("{normalized_seed_text}\0{normalized_stream_name}\0{chunk_index}");
        let chunk_value = fnv1a_64(chunk_text.as_bytes());
        let start_index = chunk_index * 8;

        seed_bytes[start_index..start_index + 8].copy_from_slice(&chunk_value.to_le_bytes());
    }

    seed_bytes
}

//----------------------------------------------------------------------------------------------------------------------
// Function: seeded_random_number_generator
//
// Description:
//
//   Create a deterministic random number generator for a named stream.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn seeded_random_number_generator(seed_text: &str, stream_name: &str) -> StdRng {

    // Feed the expanded deterministic bytes into StdRng so the caller receives a reproducible stream.

    StdRng::from_seed(deterministic_seed_bytes(seed_text, stream_name))
}

//----------------------------------------------------------------------------------------------------------------------
// Function: optional_seeded_random_number_generator
//
// Description:
//
//   Create a deterministic random number generator only when non-empty seed text is present.
//
//----------------------------------------------------------------------------------------------------------------------

pub fn optional_seeded_random_number_generator(
    seed_text: Option<&str>,
    stream_name: &str,
) -> Option<StdRng> {

    // Treat missing or blank seed text as "no deterministic generator requested."

    seed_text
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| seeded_random_number_generator(text, stream_name))
}

//----------------------------------------------------------------------------------------------------------------------
// Function: fnv1a_64
//
// Description:
//
//   Hash bytes with the deterministic FNV-1a 64-bit algorithm.
//
//----------------------------------------------------------------------------------------------------------------------

fn fnv1a_64(bytes: &[u8]) -> u64 {

    // Start from the standard FNV offset basis and mix each byte into the hash.

    let mut hash_value = FNV_OFFSET_BASIS;

    for byte in bytes {

        // Compute the FNV-1a step: xor the byte into the hash, then multiply by the FNV prime with
        // wrapping arithmetic so the result stays in 64 bits.

        hash_value ^= u64::from(*byte);
        hash_value = hash_value.wrapping_mul(FNV_PRIME);
    }

    // Return the final deterministic 64-bit hash value.

    hash_value
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    //------------------------------------------------------------------------------------------------------------------
    // Function: seed_value_is_stable
    //
    // Description:
    //
    //   Verify text seeds produce repeatable numeric seed values.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn seed_value_is_stable() {

        // Equal text should hash to the same value, while different text should not collide here.

        assert_eq!(
            deterministic_seed_value("stable seed"),
            deterministic_seed_value("stable seed")
        );
        assert_ne!(
            deterministic_seed_value("stable seed"),
            deterministic_seed_value("other seed")
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: seeded_random_number_generator_is_reproducible
    //
    // Description:
    //
    //   Verify equal seed text and stream names produce equal random streams.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn seeded_random_number_generator_is_reproducible() {

        // Build two independent generators from the same seed and stream name.

        let mut first_random_number_generator =
            seeded_random_number_generator("stable seed", "root");
        let mut second_random_number_generator =
            seeded_random_number_generator("stable seed", "root");

        // Compare consecutive draws to prove the whole stream, not only the first value, is stable.

        assert_eq!(
            first_random_number_generator.next_u64(),
            second_random_number_generator.next_u64()
        );
        assert_eq!(
            first_random_number_generator.next_u64(),
            second_random_number_generator.next_u64()
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: named_streams_are_distinct
    //
    // Description:
    //
    //   Verify stream names split one seed into independent deterministic streams.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn named_streams_are_distinct() {

        // The stream name separates consumers that share the same user-facing seed text.

        assert_ne!(
            deterministic_seed_bytes("stable seed", "root"),
            deterministic_seed_bytes("stable seed", "rollout")
        );
    }

    //------------------------------------------------------------------------------------------------------------------
    // Function: optional_seeded_random_number_generator_ignores_empty_seed_text
    //
    // Description:
    //
    //   Verify empty optional seed values do not create deterministic generators.
    //
    //------------------------------------------------------------------------------------------------------------------

    #[test]
    fn optional_seeded_random_number_generator_ignores_empty_seed_text() {

        // None and whitespace-only values mean the caller should keep using its nondeterministic RNG.

        assert!(optional_seeded_random_number_generator(None, "root").is_none());
        assert!(optional_seeded_random_number_generator(Some("   "), "root").is_none());
        assert!(optional_seeded_random_number_generator(Some("seed"), "root").is_some());
    }
}
