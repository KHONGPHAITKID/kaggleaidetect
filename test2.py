

# 2. Example Usage
original_code = "def add(a, b): return a + b"
perturbed_code = "def sum_nums(x, y): return x + y"

score_orig = get_log_likelihood(original_code)
score_pert = get_log_likelihood(perturbed_code)

print(f"Original Score: {score_orig:.4f}")
print(f"Perturbed Score: {score_pert:.4f}")
print(f"Discrepancy: {score_orig - score_pert:.4f}")