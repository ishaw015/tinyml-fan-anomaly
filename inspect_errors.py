
import numpy as np

errors = np.load("data/autoencoder_test_errors.npy")
labels = np.load("data/test_y.npy")
speeds = np.load("data/test_speed.npy")

healthy = errors[labels == 0]
faulty = errors[labels == 1]

print("HEALTHY RECONSTRUCTION ERRORS")
print("Mean:", healthy.mean())
print("Median:", np.median(healthy))
print("90th percentile:", np.percentile(healthy, 90))
print("Maximum:", healthy.max())

print("\nFAULTY RECONSTRUCTION ERRORS")
print("Mean:", faulty.mean())
print("Median:", np.median(faulty))
print("10th percentile:", np.percentile(faulty, 10))
print("Minimum:", faulty.min())

print("\nFAULTY ERROR BY SPEED")
for low, high in [
    (10, 20), (20, 30), (30, 40),
    (40, 50), (50, 60), (60, 70)
]:
    mask = (
        (labels == 1) &
        (speeds >= low) &
        (speeds < high)
    )

    if mask.any():
        print(
            f"{low}-{high} Hz:",
            f"mean={errors[mask].mean():.4f},",
            f"median={np.median(errors[mask]):.4f}"
        )


print("\nHEALTHY ERROR BY SPEED")

for low, high in [
    (10, 20), (20, 30), (30, 40),
    (40, 50), (50, 60), (60, 70)
]:
    mask = (
        (labels == 0) &
        (speeds >= low) &
        (speeds < high)
    )

    if mask.any():
        print(
            f"{low}-{high} Hz:",
            f"mean={errors[mask].mean():.4f},",
            f"median={np.median(errors[mask]):.4f}"
        )
