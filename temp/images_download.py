## download images folder in original folder
for x in range(99,100):
  !rsync --recursive --verbose rsync://78.46.86.149:873/danbooru2019/original/00{x} "/content/drive/My Drive/dataSet/danbooru2019/original"
  
## unzip metadate file
!tar -xvf  '/content/drive/My Drive/dataSet/danbooru2019/metadata.json.tar.xz' -C '/content/drive/My Drive/dataSet/danbooru2019/metadata'