import requests
import json

def write(line):
    o = open("output.txt", "a")
    o.write(string)
    o.close()

f = open(".\\..\\Dane\\movie.csv")


for line in f:
    number = line.split(";")[1]
    url = "https://api.themoviedb.org/3/movie/" + str(number)+ "?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzOTAzMmEwMmQ5Y2E4YmJmNzdmNWUwMTBlZjg1ZWRkOCIsIm5iZiI6MTcyOTkyMTU0My40NTIxNzYsInN1YiI6IjY3MWM4MGFjNmU0MjEwNzgwZjc5MTc1YyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.qBgR_2wIpK0dcHQ4L31MD6zzGFpMeKK3G3h3jW9E-Q0"
    }

    response = requests.get(url, headers=headers)
    j = json.loads(response.text)
    searched_els = ['budget', 'popularity', 'vote_average']
    string = str(number) + ";"
    part = ""
    for el in searched_els:
        string = string + str(j[el]) + ";"
    
    for one in j["genres"]:
        part = part + one["name"] + "#"

    string = string + part[:len(part)-1] + ";"
    part = ""
    for one in j["production_companies"]:
        part = part + one["origin_country"] + "#"

    string = string + part[:len(part)-1] + ";"

    part = ""
    for one in j["origin_country"]:
        part = part + one + "#"

    string = string + part[:len(part)-1] + "\n"
    write(string)
    
    
    #print(j["genres"][1]["name"])
    #print(j["production_companies"][1]['origin_country'])


f.close()



    
