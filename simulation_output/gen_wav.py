"""Generate a linear-chirp WAV (220 Hz -> 3520 Hz, 3 s, 44100 Hz, mono PCM16)."""
import sys, wave, struct, math

def main():
    out = sys.argv[1]
    SR, DUR = 44100, 3.0
    F0, F1  = 220.0, 3520.0
    N = int(SR * DUR)

    samples = []
    for i in range(N):
        t     = i / SR
        phase = 2.0 * math.pi * (F0 * t + (F1 - F0) * t * t / (2.0 * DUR))
        samples.append(int(32767 * 0.5 * math.sin(phase)))

    with wave.open(out, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(struct.pack(f'<{N}h', *samples))

    print(f'  WAV: {out}  ({N} samples @ {SR} Hz)')

if __name__ == '__main__':
    main()
