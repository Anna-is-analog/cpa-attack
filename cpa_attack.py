# cpa_attack.py
# Correlation Power Analysis (CPA) attack on AES
# works in two modes:
#   - simulation mode: generates fake traces (no hardware needed)
#   - real mode: captures traces from ChipWhisperer Lite


import numpy as np
import matplotlib.pyplot as plt

USE_REAL_HW = False  # true when using real hardware

# -------------------------------------------------------
# AES S-box (needed to predict power consumption)
# -------------------------------------------------------
SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

def hamming_weight(x):
    # count number of 1 bits -- our power consumption model
    # assumption: more 1s toggling in the circuit = more power drawn
    return bin(x).count('1')

# -------------------------------------------------------
# trace collection -- simulation mode
# -------------------------------------------------------
def collect_traces_simulated(true_key_byte, num_traces=1000, num_samples=50):
    print(f"collecting {num_traces} simulated traces...")

    plaintexts = np.random.randint(0, 256, num_traces, dtype=np.uint8)
    traces = np.zeros((num_traces, num_samples))

    for i, pt in enumerate(plaintexts):
        sbox_out = SBOX[pt ^ true_key_byte]
        real_power = hamming_weight(sbox_out)

        # mostly noise, with the real signal leaking at sample 10
        noise = np.random.normal(0, 1.5, num_samples)
        traces[i] = noise
        traces[i][10] += real_power

    return plaintexts, traces

# -------------------------------------------------------
# trace collection -- real ChipWhisperer Lite mode
# -------------------------------------------------------
def collect_traces_real(num_traces=2000):
    import chipwhisperer as cw

    print("connecting to ChipWhisperer Lite...")
    scope = cw.scope()
    target = cw.target(scope)

    scope.default_setup()
    scope.adc.samples = 500  # increase this if missing the leakage window

    plaintexts = []
    traces = []

    print(f"capturing {num_traces} traces...")
    for i in range(num_traces):
        pt = cw.bytearray(np.random.randint(0, 256, 16).tolist())
        trace = cw.capture_trace(scope, target, pt)

        if trace is not None:
            plaintexts.append(pt[0])  # we're attacking the first key byte
            traces.append(trace.wave)

        if i % 100 == 0:
            print(f"  {i}/{num_traces} traces captured")

    scope.dis()
    target.dis()

    print("done capturing.")
    return np.array(plaintexts, dtype=np.uint8), np.array(traces)

# -------------------------------------------------------
# CPA attack
# for each of the 256 possible key guesses:
#   predict what the power should have been using hamming weight
#   correlate that prediction against real traces at every sample point
#   the guess with the highest correlation peak is the real key
# -------------------------------------------------------
def cpa_attack(plaintexts, traces):
    num_traces, num_samples = traces.shape
    print(f"running CPA on {num_traces} traces x {num_samples} samples...")

    correlations = np.zeros((256, num_samples))

    for key_guess in range(256):
        predictions = np.array([
            hamming_weight(SBOX[pt ^ key_guess]) for pt in plaintexts
        ], dtype=float)

        for sample in range(num_samples):
            corr = np.corrcoef(predictions, traces[:, sample])[0, 1]
            correlations[key_guess, sample] = abs(corr)

        if key_guess % 64 == 0:
            print(f"  tested {key_guess}/256 key guesses...")

    return correlations

# -------------------------------------------------------
# find the winning key guess
# -------------------------------------------------------
def find_key(correlations):
    max_corr_per_key = correlations.max(axis=1)
    best_key = np.argmax(max_corr_per_key)
    best_corr = max_corr_per_key[best_key]
    return best_key, best_corr

# -------------------------------------------------------
# plot -- all 256 guesses in gray, correct key in red, recovered in blue
# -------------------------------------------------------
def plot_results(correlations, true_key, recovered_key):
    plt.figure(figsize=(12, 5))

    for k in range(256):
        if k != true_key:
            plt.plot(correlations[k], color='lightgray', linewidth=0.5, alpha=0.5)

    plt.plot(correlations[true_key], color='red', linewidth=1.5,
             label=f'correct key (0x{true_key:02X})')
    plt.plot(correlations[recovered_key], color='blue', linewidth=1, linestyle='--',
             label=f'recovered key (0x{recovered_key:02X})')

    plt.xlabel('sample point in trace')
    plt.ylabel('correlation coefficient')
    plt.title('CPA attack -- correlation vs sample point for all 256 key guesses')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cpa_result.png', dpi=150)
    plt.show()
    print("plot saved as cpa_result.png")

# -------------------------------------------------------
# main
# -------------------------------------------------------
if __name__ == "__main__":

    if USE_REAL_HW:
        # real mode -- plug in CW-Lite 
        # we don't know the true key here -- that's the whole point
        plaintexts, traces = collect_traces_real(num_traces=2000)
        TRUE_KEY = None
    else:
        # simulation mode -- no hardware needed, good for testing the analysis code
        TRUE_KEY = 0xBE
        plaintexts, traces = collect_traces_simulated(true_key_byte=TRUE_KEY, num_traces=1000)

    # run the attack
    correlations = cpa_attack(plaintexts, traces)

    # recover the key
    recovered_key, confidence = find_key(correlations)

    print(f"\n--- results ---")
    if TRUE_KEY is not None:
        print(f"true key byte:      0x{TRUE_KEY:02X}")
    print(f"recovered key byte: 0x{recovered_key:02X}")
    print(f"peak correlation:   {confidence:.4f}")
    if TRUE_KEY is not None:
        print(f"attack {'SUCCEEDED' if recovered_key == TRUE_KEY else 'FAILED'}")

    # if we don't know the true key just highlight the recovered one
    plot_key = recovered_key if TRUE_KEY is None else TRUE_KEY
    plot_results(correlations, plot_key, recovered_key)
