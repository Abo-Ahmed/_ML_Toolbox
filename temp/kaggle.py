from google.colab import files
files.upload()

!pip install -q kaggle
!pip install -q kaggle-cli
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets list -s tagged-anime-illustrations
!kaggle datasets download -d mylesoneill/tagged-anime-illustrations

!unzip tagged-anime-illustrations.zip
!unzip danbooru-metadata.zip
