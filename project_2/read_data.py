import re
import urllib.request

f_m = open("facescrub_actors.txt","r")
f_w = open("facescrub_actresses.txt","r")
i = 0

for l in f_m.readlines():
    x = re.search("http[^\s]+", l)
    print(l)
    if x != None:
        try:
            url = x.group()
            urllib.request.urlretrieve(url, "imgs/" + str(i) + ".jpg")
            i += 1
        except:
            print("deu ruim")
            pass

f_m.close()
f_w.close()
