def get_rating(id , json_metadata):
  for item in json_metadata :
    if item['id'] == id :
      return item['rating']
  return -1

