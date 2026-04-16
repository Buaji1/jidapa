import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle

# โหลดชุดข้อมูล Iris
iris = load_iris()

# X คือข้อมูลฟีเจอร์ (Features) เช่น ความยาวและความกว้างของกลีบ
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# y คือเป้าหมาย (Target) หรือสายพันธุ์ของดอกไม้
y = iris.target

# แบ่งข้อมูลเป็นส่วน Train (70%) และ Test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างโมเดล Logistic Regression (กำหนด max_iter=200 เพื่อให้มันคำนวณจนจบกระบวนการ)
model = LogisticRegression(max_iter=200)

# สั่งให้โมเดลเริ่มเรียนรู้ (Train)
model.fit(X_train, y_train)

print("สอน AI สำเร็จแล้ว!")

# กำหนดชื่อไฟล์และบันทึกโมเดล
filename = 'iris_model.pkl'
pickle.dump(model, open(filename, 'wb'))

print(f"บันทึกไฟล์โมเดลสำเร็จในชื่อ {filename}")
