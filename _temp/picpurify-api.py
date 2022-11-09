!pip install requests
import requests

picpurify_url = 'https://www.picpurify.com/analyse.php'
result_url = requests.post(picpurify_url,data = {"url_image":"https://i.ytimg.com/vi/AGBjI0x9VbM/maxresdefault.jpg", "API_KEY":"XXX", "task":"porn_detection,drug_detection,gore_detection", "origin_id":"xxxxxxxxx", "reference_id":"yyyyyyyy"})
print (str(result_url.content) + '\n')
