import os
import shutil
import random

# Original dataset
SOURCE = "Dataset"

# New split folders
OUTPUT = "dataset_split"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# Create train, validation and test folders
for split in ["train", "val", "test"]:
	os.makedirs(os.path.join(OUTPUT, split), exist_ok=True)

# Go through every class
classes = os.listdir(SOURCE)

for class_name in classes:

	class_path = os.path.join(SOURCE, class_name)

	if not os.path.isdir(class_path):
		continue

	# Get image files
	images = [
		f for f in os.listdir(class_path)
		if f.lower().endswith((".jpg", ".jpeg", ".png"))
	]

	random.shuffle(images)

	total = len(images)

	train_end = int(total * TRAIN_RATIO)
	val_end = train_end + int(total * VAL_RATIO)

	train_images = images[:train_end]
	val_images = images[train_end:val_end]
	test_images = images[val_end:]

	# Create class folders
	for split in ["train", "val", "test"]:
		os.makedirs(
			os.path.join(OUTPUT, split, class_name),
			exist_ok=True
		)

	# Copy images
	for image in train_images:
		shutil.copy2(
			os.path.join(class_path, image),
			os.path.join(OUTPUT, "train", class_name, image)
		)

	for image in val_images:
		shutil.copy2(
			os.path.join(class_path, image),
			os.path.join(OUTPUT, "val", class_name, image)
		)

	for image in test_images:
		shutil.copy2(
			os.path.join(class_path, image),
			os.path.join(OUTPUT, "test", class_name, image)
		)

	print(
		f"{class_name}: "
		f"Train={len(train_images)}, "
		f"Val={len(val_images)}, "
		f"Test={len(test_images)}"
	)

print("\nDataset splitting completed!")
