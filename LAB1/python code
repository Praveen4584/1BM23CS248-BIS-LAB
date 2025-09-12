import random
import numpy as np
from PIL import Image, ImageEnhance
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from skimage import img_as_float
# ---------- Parameters ----------
POP_SIZE = 20
N_GEN = 15
MUT_RATE = 0.3
ELITE = 2  # keep best individuals each gen

# Load image
original = Image.open("input.jpg").convert("RGB")
original_np = np.array(original)

# If you have a reference image, load it
try:
    target = Image.open("target.jpg").convert("RGB")
    target_np = np.array(target)
    USE_REFERENCE = True
except:
    USE_REFERENCE = False

# ---------- Apply Adjustments ----------
def apply_adjustments(params):
    b, c, s = params
    img = original.copy()
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Sharpness(img).enhance(s)
    return img

# ---------- Fitness Function ----------
def fitness(params):
    img = apply_adjustments(params)
    img_np = np.array(img)

    if USE_REFERENCE:
        return ssim(target_np, img_np, channel_axis=2)
    else:
        gray = img.convert("L")
        gray_np = img_as_float(np.array(gray))
        entropy = -np.sum(gray_np * np.log2(gray_np + 1e-10))
        edge_strength = np.mean(sobel(gray_np))
        return entropy + edge_strength

# ---------- GA Core ----------
def init_population(size):
    return [np.array([
        random.uniform(0.5, 2.0),  # Brightness
        random.uniform(0.5, 2.0),  # Contrast
        random.uniform(0.5, 2.0)   # Sharpness
    ]) for _ in range(size)]

def crossover(p1, p2):
    alpha = random.random()
    return alpha * p1 + (1 - alpha) * p2

def mutate(ind):
    if random.random() < MUT_RATE:
        ind += np.random.normal(0, 0.2, size=3)
        ind = np.clip(ind, 0.5, 2.0)
    return ind

def select(pop, fitnesses):
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if fitnesses[i] > fitnesses[j] else pop[j]

# ---------- Run GA ----------
def run_ga():
    pop = init_population(POP_SIZE)

    for gen in range(N_GEN):
        fitnesses = [fitness(ind) for ind in pop]
        ranked = sorted(zip(pop, fitnesses), key=lambda x: x[1], reverse=True)
        best_ind, best_fit = ranked[0]

        print(f"Gen {gen}: Best fitness = {best_fit:.4f}, Params = {best_ind}")

        new_pop = [ind.copy() for ind, _ in ranked[:ELITE]]

        while len(new_pop) < POP_SIZE:
            p1, p2 = select(pop, fitnesses), select(pop, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        pop = new_pop

    best_img = apply_adjustments(best_ind)
    best_img.save("optimized.jpg")
    print("Best parameters found:", best_ind)
    display(best_img)  # Show inside Jupyter

# Run it
run_ga()
