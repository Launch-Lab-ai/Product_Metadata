import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Step 1: Setup ===
chunksize = 100000
bucket_counts = {}
uncategorized_samples = []
os.makedirs('./data/amazon', exist_ok=True)
os.makedirs('./figures/amazon', exist_ok=True)

# === Step 2: Define keyword sets ===
bucket_keywords = {
    'Clothing': [
        'anarkali', 'apparel', "boy's", 'blouse', 'boots', 'casual shoes',
        'dhoti', 'dress', 'ethnic', 'fashionwear', 'footwear', 'girl\'s',
        'jeans', 'jacket', 'kurta', 'kurti', 'leggings', 'lingerie',
        'long sleeve', 'loungewear', 'nightwear', 'pant', 'pajama',
        'peplum', 'polo', 'pyjama', 'sandal', 'sandals', 'shirt',
        'shorts', 'socks', 'stylish', 'sweater', 'sweatshirt', 't-shirt',
        'top', 'trousers', 'tunic', 'underwear', 'sleeveless'
    ],
    'Jewelry': [
        'analog watch', 'anklet', 'bracelet', 'choker', 'chronograph',
        'cubic zirconia', 'digital watch', 'diamond', 'earring', 'gold',
        'jewellery', 'jewelry', 'necklace', 'nose pin', 'pendant',
        'quartz', 'ring', 'sapphire crystal', 'silver', 'sunglasses',
        'watch', 'wristwatch'
    ],
    'Tech Gadgets': [
        'adapter', 'amplifier', 'bluetooth', 'cable', 'case', 'connector',
        'cctv', 'electronics', 'gadget', 'gimbal', 'hdmi', 'headphones',
        'lcd', 'mobile cover', 'power bank', 'samsung', 'smartphone',
        'speaker', 'surveillance', 'tablet', 'tech', 'tripod', 'usb',
        'wire'
    ],
    'Home Decor': [
        'bed sheet', 'blanket', 'coasters', 'cookware', 'cushion',
        'cutlery', 'decor', 'dish rack', 'doormat', 'furnishing',
        'idol', 'kitchen', 'lamp', 'laundry', 'mat', 'mirror', 'organizer',
        'photo frame', 'pillow', 'planter', 'quilt', 'reading pillow',
        'statue', 'storage box', 'table runner', 'tablecloth', 'vase',
        'wall art', 'wall sticker' , 'jhula' , 'artwall'
    ],
    'Personal Care': [
        'activated charcoal', 'bb cream', 'conditioner', 'cosmetic',
        'cream', 'eyeliner', 'face mask', 'face wash', 'kaolin',
        'lip balm', 'makeup remover', 'maybelline', 'moisturizer',
        'roll on', 'scrub', 'serum', 'shampoo', 'skincare', 'soap',
        'toothbrush', 'zinc oxide'
    ],
    'Footwear': [
        'boot', 'brogue', 'cleats', 'flip flop', 'footbed', 'jooti',
        'loafer', 'moccasin', 'oxford', 'sandal', 'shoe', 'slipper',
        'sneaker', 'nike'
    ],
    'Furniture & Fixtures': [
        'bench', 'bookshelf', 'cabinet', 'chair', 'coffee table',
        'desk', 'dresser', 'drawer', 'furniture', 'nightstand',
        'ottoman', 'rack', 'sofa', 'stool', 'table', 'vanity'
    ],
    'Books & Media': [
        'biography', 'book', 'chronicle', 'guidebook', 'hardcover',
        'lesson', 'literature', 'memoir', 'novel', 'reading',
        'reading comprehension', 'renditions', 'sheet music',
        'storybook', 'textbook'
    ],
    'Gifts': [
        'anniversary', 'birthday', 'collectible', 'customized',
        'decorative box', 'figurine', 'gift', 'handbag', 'keychain',
        'mug', 'patch', 'plush', 'shot glass', 'souvenir', 'sticker',
        'vinyl figure', 'wallet' , 'hot wheels', 'picture frame'
    ],
    'Auto Industrial': [
        'belt', 'brake', 'car cover', 'clevis', 'cnc', 'drill bit',
        'engine', 'fuse', 'garage', 'milling', 'motor', 'saddle',
        'seat cover', 'timing belt', 'tool', 'transistor', 'vehicle'
    ]
}



# === Step 3: Assign best-matching bucket ===
def assign_amazon_bucket(row):
    text = ' '.join([
        str(row.get('TITLE', '')), str(row.get('DESCRIPTION', '')), str(row.get('BULLET_POINTS', ''))
    ]).lower()

    scores = {bucket: sum(kw in text for kw in keywords) for bucket, keywords in bucket_keywords.items()}
    best_bucket = max(scores, key=scores.get)
    return best_bucket if scores[best_bucket] > 0 else 'Uncategorized'

# === Step 4: Process CSV in chunks ===
output_file = './data/amazon/amazon_buckets.csv'
first_write = True

for i, chunk in enumerate(pd.read_csv('./data/amazon/amazon_cleaned.csv', chunksize=chunksize)):
    chunk['final_bucket'] = chunk.apply(assign_amazon_bucket, axis=1)

    # Collect sample uncategorized
    if len(uncategorized_samples) < 40:
        uncats = chunk[chunk['final_bucket'] == 'Uncategorized']
        if not uncats.empty:
            needed = 40 - len(uncategorized_samples)
            uncategorized_samples.extend(uncats[['TITLE', 'DESCRIPTION', 'BULLET_POINTS']].head(needed).to_dict('records'))

    # Save chunk output
    chunk.to_csv(output_file, mode='w' if first_write else 'a', header=first_write, index=False)
    first_write = False

    # Update and show bucket counts
    counts = chunk['final_bucket'].value_counts().to_dict()
    for label, count in counts.items():
        bucket_counts[label] = bucket_counts.get(label, 0) + count

    print(f"✅ Chunk {i+1}/~23 processed — counts: {counts}")

# === Step 5: Final Bucket Summary ===
print("\n📦 Final Bucket Counts:")
for label, count in bucket_counts.items():
    print(f"➡️  {label:18s}: {count}")

# # === Step 6: Sample 'Uncategorized' Items ===
# print("\n🔎 Sample 'Uncategorized' Products (40 examples):\n")
# for i, row in enumerate(uncategorized_samples):
#     print(f"🟦 {i+1}.")
#     print(f"🔹 TITLE      : {row['TITLE']}")
#     print(f"📝 DESCRIPTION: {row['DESCRIPTION']}")
#     print(f"📌 BULLETS    : {row['BULLET_POINTS']}\n{'-'*80}")

# === Step 7: Visualization ===
plt.figure(figsize=(10, 6))
labels = list(bucket_counts.keys())
values = list(bucket_counts.values())

bars = plt.bar(labels, values, color=['#5DADE2', '#EC7063', '#58D68D', '#A569BD', '#95A5A6', '#F5B041', '#76D7C4', '#DC7633', '#D98880', '#7DCEA0'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 100, f'{yval}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Product Count per Bucket", fontsize=16)
plt.xlabel("Bucket", fontsize=12, fontweight='bold')
plt.ylabel("Number of Products", fontsize=12, fontweight='bold')
plt.xticks(rotation=15, fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.savefig('./figures/amazon/bucket_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
