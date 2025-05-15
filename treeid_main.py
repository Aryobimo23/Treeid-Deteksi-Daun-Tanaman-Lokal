import joblib
from extract_features import extract_leaf_features

model = joblib.load('model/treeid_model.pkl')

def classify_leaf(image_path):
    features = extract_leaf_features(image_path)
    if not features:
        print("Gagal mengekstrak fitur dari gambar.")
        return
    prediction = model.predict([features])
    print(f"Jenis pohon terdeteksi: {prediction[0]}")

if __name__ == "__main__":
    img_path = input("Masukkan path gambar daun: ")
    classify_leaf(img_path)