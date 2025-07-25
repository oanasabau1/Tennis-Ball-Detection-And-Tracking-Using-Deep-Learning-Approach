import requests

url = ("https://drive.usercontent.google.com/download?id=1lhAaeQCmk2y440PmagA0KmIVBIysVMwu&export=download&authuser=0"
       "&confirm=t&uuid=3077628e-fc9b-4ef2-8cde-b291040afb30&at=APZUnTU9lSikCSe3NqbxV5MVad5T%3A1708243355040")
headers = {
    "Host": "drive.usercontent.google.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 "
                  "Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
              "application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Cookie": "HSID=Ag2OIHvsd2Wub4C7z; SSID=AWnBcQKwDHiTrZAU1; APISID=pltrFZgE9lJ0o1gq/AN9feEHYvs8oHd519; "
              "SAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; __Secure-1PAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; "
              "__Secure-3PAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; "
              "SID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI-uttBbVDRolhF"
              "-hY16nwHXw0gACgYKAWISAQASFQHGX2MivNTw_E_toJuIRy6LMpKNOBoVAUF8yKpFSmvq7AMjvEWeNc50Zff40076; "
              "__Secure-1PSID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI"
              "-utbSY2jBY1VXuw8gYl5hIO2QACgYKAXsSAQASFQHGX2MihVCJ1PwLozGqZgdSatM9QhoVAUF8yKpgrsTvI8i_UE-YHpoN7Gx-0076"
              "; __Secure-3PSID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI"
              "-utwVfPl2imdPimZJ9tdDZGQAACgYKAUESAQASFQHGX2MiEJ49mV4jME2kttDAV5hwWBoVAUF8yKp80mIgju1lu-q4nI7VsFDM0076"
              "; "
              "NID=511"
              "=efI9IZpxtyJ7Dw1MAUXU8FlzS5jXGewY4Er8HliWc3A0RSWdgvNDyKY66ETjgRyTGWPbWODSmiSeYSBab5SPHVwqbJxd6ZeGW2f6BkHi61UKksXPH0CVJRM1hKpMjHPU5qw7tboM2Mi87NrosV8COB-GCLulLLbjOoSAEQewTe8NVZ5Owq8IkwvxFGfJkmUKEMkFWrw9yb5nTDl3wbZEsGFI92iEdNTSxSRovNCIPN2US-SCFdQ0m2BtvwdiWZbgnn7dSQ8yPA145Kk2BA-ATpJNJ6SJHEHLQY-9CPail9D5qgJgxR925EUg5RGCpEu9wS5xbA62KTa19wAvbAq7Dk3TWc-iX4p1s7ESFyDC7yMpFxiFPJjqkWwFi_ZfiK2TW2t0TQ60DFBxqOytQaLyHrkEvD-CQPVj6OCOP22cZY0Cu61HaAQgFO9pXH-kJUlywzVdbirJumN5gswyaQ49b3KdLcG0Jb7brOMTM24T2nGtQ10hJzsnTwX7dBk3ujqQrI_DGuURvPassPUrIZ0; AEC=Ae3NU9MOEGeKAZjP6INpOYbyMraWAWztmx5pJB_1ILu1furiTy1K37k15u0; __Secure-1PSIDTS=sidts-CjEBYfD7Z9twEKTWJ9gU7KG-rLbxJGNRQIoG3wH6JVu6yiCC2fsRrm7tN8L6d5WlILrnEAA; __Secure-3PSIDTS=sidts-CjEBYfD7Z9twEKTWJ9gU7KG-rLbxJGNRQIoG3wH6JVu6yiCC2fsRrm7tN8L6d5WlILrnEAA; 1P_JAR=2024-02-18-08; SIDCC=ABTWhQExCxkfmwCkG1RaEgz8U1ZkPeh3HmLMUdMt8S5cNSsLY5U5rAL6wlvq7dtjRw7zrtAbqsFI; __Secure-1PSIDCC=ABTWhQH0jLeRIS6Tu3LS8DXB5Q3gGDq9LTmlk60FKu795Bf0UbzsOcYWVAE96clq5aAL8i724Q0; __Secure-3PSIDCC=ABTWhQHIFcyv3nZYwp78WXEQal71jCE_ZsGT5lXs8VLr7XDIfFqHcLTIPz4HxzJb9ZnYQ5l2s9eU",
    "Connection": "keep-alive"
}

response = requests.get(url, headers=headers, stream=True)
with open("tennis_court_det_dataset.zip", "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print("Download complete.")