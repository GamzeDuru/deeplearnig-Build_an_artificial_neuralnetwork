
#Farklı veri türleri ile işlem yapabilir.Bu sebeple pandas tercih edilmektedir.
#Seriler(sıralı liste gibi) && dataFrame(çerçeve):excel doyası gibi ekrana çıktı verir ve farklı veri tiplerini kullanabilir.
#Ayrıca .txt gibi kolayca veri alabilir.
import pandas as pd

#veri görselleştirmesinde kullanılan temel kütüphane
import matplotlib.pyplot as plt

#Verilerimizi birebir sayısallaştırmak için kullanılan fonksiyondur :label encoder
# özellik ölçeklendirme:minmax scala
#preprocessing (önişleme)
#sklearn.preprocessing paketi birkaç ortak yardımcı işlev sağlar, transformatör sınıfları(hızlı), ham özellik vektörlerini aşağı yöndeki tahminciler için daha uygun bir gösterime dönüştürür.
#Transformatör sınıfı:
#from .. importifade
#İfade, from..import her şeyi içe aktarmak yerine bir modülden belirli işlevleri/değişkenleri içe aktarmanıza olanak tanır.
from sklearn.preprocessing import LabelEncoder , minmax_scale


#SKLEARN:Sınıflandırma, kümeleme ve model seçimi için kullanılabilecek veri işlemeye yönelik çeşitli özellikler sunan bir Python kütüphanesidir .
#Model_selection, verileri analiz etmek için bir plan oluşturmak ve ardından bunu yeni verileri ölçmek amacıyla kullanılan bir yöntemdir . Uygun bir model seçmek, tahmin yaparken doğru sonuçlar üretmenizi sağlar .
# train_test_split veri dizilerini iki alt kümeye bölmek için Sklearn model seçiminde bir işlevdir : eğitim verileri ve test verileri için. Bu fonksiyon sayesinde veri setini manuel olarak bölmenize gerek kalmaz
from sklearn.model_selection import train_test_split

#Excel dosyasındaki verileri okuyoruz.
data=pd.read_excel('C:\\Users\Gamze\Desktop\listem.xlsx')
#C:\\Users\Gamze\Downloads\liste.xlsx
#dosyadaki ilk 5 satırı okur(deger vermezsek)
print(data.head())

#kaç satır,kaç sütun
print(data.shape)

print(data['Class'].unique())

#drop:sütun ya da satırı kaldırır
# axis=1 ifadesi sütunu silmek isteğimizi belirtmektedir.axis=0 satır silmek istediğimizi beliritir.
X=data.drop('Class', axis=1)
print(X)

#loc komutu ile etiket kullananarak verimize ulaşırız
# loc parametresinin satır etiketini : olarak vermek tüm satırlar demektir.
y=data.loc[:,'Class']
print(y)

lb=LabelEncoder()
y=lb.fit_transform(y)

#Değişken içinde yer alan sayıları genellikle 0 ve 1 arasına hapseden bir yöntemdir.
X_scaled=minmax_scale(X)
X=pd.DataFrame(X_scaled)
print(X.head())

print(y)

X_train, X_temporary, y_train, y_temporary = train_test_split(X, y, train_size=0.8)
X_val, X_test, y_val, y_test =train_test_split(X_temporary, y_temporary, train_size=0.5)

print(f'Length of the dataset: {len(X)}')
print(f'Length of the dataset_train: {len(X_train)}')
print(f'Length of the dataset_val: {len(X_val)}')
print(f'Length of the dataset_test: {len(X_test)}')


print(y_train)
print(X_train)


# makine öğrenimi için ücretsiz ve açık kaynaklı bir yazılım kütüphanesidir
# derin sinir ağlarının eğitimi ve çıkarımına özel olarak odaklanır

'''TensorFlow, makine öğrenimi ve derin öğrenmeye yönelik uçtan uca açık kaynaklı bir kütüphanedir. 
En büyük avantajlarından biri, derin öğrenme algoritmalarının gerektirdiği ağır hesaplamaların 
GPU'lar üzerinde yapılabilmesidir. GPU'lar, yoğun hesaplamaları bilgisayarınızın Merkezi İşlem Birimleri
olan CPU'lardan çok daha hızlı gerçekleştirebilen Grafik İşlem Birimleridir. Elbette CPU'ları kullanmak
hala mümkün ancak bu çok uzun bir zaman alacaktır.'''
import tensorflow as tf

#Keras library:yapay sinir ağları oluşturmaya yardımcı olur.
from tensorflow import keras

#Keras'ın, bir model nesnesine kolayca katmanlar, aktivasyon fonksiyonları vb. eklememizi sağlayan Sequential API adında bir modülü vardır.
model = tf.keras.Sequential()

#giriş katmanını oluşturuyoruz
#if giris<0 :0 else:girilen degeri döndürür (Relu:aktivasyon fonksiyonumuz)
#ilk argüman gizli katmanda istediğimiz düğüm sayısını belirtir(4)
#ikinci argüman sütun sayısını belirtir
input_layer = tf.keras.layers.Dense(4096, input_shape=(34,) , activation = 'relu')
model.add(input_layer)


#gizli katmanları oluşturuyoruz
#model için en uygun gizli katman ve düğüm sayısı hakkında bir tahminde bulunmak zordur. 
#Bu nedenle rakamları rastgele seçip performansa bakıyoruz ve gerekirse daha sonra ayarlamalar yapıyoruz.
#her biri 4 adet düğüme sahip 2 adet katman oluşturuyoruz.
#Bilgisayarlar ikili sayılarla çalıştığından eğitim hızının artması için 2'nin kuvvetleri olan katman sayısının seçilmesi önemle tavsiye edilir
#Bırakma(Dropout), en basit ifadeyle sinir ağlarında aşırı uyumu önlemek için kullanılır.
#düğümler rastgele seçilir ve "kapatılacak" düğümlerin sayısı, bırakma oranına göre seçilir.
#bırakma oranını 0,5 olarak ayarladığımız için düğümlerin yarısı kapatılacaktır.


#first hidden layer
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

#second hidden layer
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

#third hidden layer
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

#fourth hidden layer
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

#add the output layer

'''
 Düğüm sayısı hedef sınıf sayısına eşit olacaktır.
 İkiden fazla sınıfımız olduğundan çıkış katmanında softmax aktivasyon fonksiyonunu kullanacağız.
 Softmax işlevi sayıları olasılıklara dönüştürür. 
 Vektör üzerindeki her değerin ölçeğine göre bir olasılık oranı verir. Giriş değerleri negatif, 
 pozitif veya sıfır olabilir, fark etmez, softmax fonksiyonu 0 ile 1 arasındaki her değeri dönüştürecektir.
 Çok sınıflı sınıflandırma problemlerinde yaygın olarak sinir ağlarının çıkış katmanında kullanılır.
 Çünkü bu gibi sınıflandırma problemlerinde çıktıların her sınıf için olasılık olarak temsil edilmesine 
 ihtiyacımız var.
'''
model.add(tf.keras.layers.Dense(7, activation='softmax'))


#modeli derleme
'''
 . Eğitime yönelik modeli yapılandırmak için derleme yöntemini kullanacağız. 
 Derleme yönteminin içinde kullanmak istediğimiz optimize ediciyi ve kayıp fonksiyonunu tanımlamamız gerekir.
 Optimize edici için “Adam”ı ve kayıp fonksiyonu için “Seyrek Kategorik Çapraz Entropi (Sparse Categorical Cross Entropy)”yi kullanacağız.
 Adam optimizer, stokastik gradyan inişinin ve derin öğrenme modelleri için "git" algoritmasının genişletilmiş bir versiyonudur.
 Bunun nedeni, Adam optimize edicinin sonuçlarının genel olarak diğer tüm optimizasyon algoritmalarından daha iyi olması,
 daha hızlı hesaplama süresine sahip olması ve ayarlama için daha az parametre gerektirmesidir. Bu yüzden genellikle denemelere bununla başlarız
 ve model kötü performans gösterirse değiştiririz. Ayrıca, bu kayıp fonksiyonunu seçtik çünkü tamsayı etiketli çok sınıflı sınıflandırma problemleri için en uygun olanıdır.

'''

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy' , metrics=['accuracy'])

#train the model 100 epochs

results=model.fit(X_train, y_train, epochs=100, validation_data= (X_val, y_val))


#Grafik çizdirme
plt.plot(results.history['loss'], label='Train')
plt.plot(results.history['val_loss'], label='Test')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()



'''Kayıp olabildiğince 0'a yakın olmalı
Doğruluk ise olabildiğince 1'e yakın olmalıdır.
Grafikteki çizgiler birbirrinden çok uzaklaşmamıştır. Bu istenilen durumdur.

'''
#doğruluk ve kayıp degerini ekrana yazdırma
test_result=model.test_on_batch(X_test,y_test)
print(test_result)


#Modeli daha iyi eğitmek için farklı yöntemler vardır. Örneğin hiperparametreleri değiştirebiliriz.











