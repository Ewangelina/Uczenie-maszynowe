import requests
import json

# Function to write to the output file
def write(line):
    # Open the file with UTF-8 encoding to handle any special characters
    with open("output.txt", "a", encoding="utf-8") as o:
        o.write(line)

# Replace 'YOUR_ACCESS_TOKEN' with your actual TMDB API Bearer token
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzOTAzMmEwMmQ5Y2E4YmJmNzdmNWUwMTBlZjg1ZWRkOCIsIm5iZiI6MTcyOTkyMTU0My40NTIxNzYsInN1YiI6IjY3MWM4MGFjNmU0MjEwNzgwZjc5MTc1YyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.qBgR_2wIpK0dcHQ4L31MD6zzGFpMeKK3G3h3jW9E-Q0"

# Open the input CSV file for reading
with open("Dane\\movie.csv") as f:
    for line in f:
        # Get the movie ID from the CSV file
        number = line.strip().split(";")[1]
        url = f"https://api.themoviedb.org/3/movie/{number}?language=en-US"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "accept": "application/json"
        }

        # Send a request to the TMDB API
        response = requests.get(url, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            j = response.json()
            
            # Prepare output line with the movie ID
            output_line = f"{number};"
            
            # All possible fields that you can extract from the movie object
            fields = {
                'budget': 'N/A', 
                'popularity': 'N/A', 
                'vote_average': 'N/A',
                'title': 'N/A', 
                'release_date': 'N/A', 
                'runtime': 'N/A', 
                'original_language': 'N/A',
                'vote_count': 'N/A',
                'status': 'N/A',
                'tagline': 'N/A',
                'overview': 'N/A',
                'genres': 'N/A',
                'production_companies': 'N/A',
                'production_countries': 'N/A',
                'spoken_languages': 'N/A',
                'revenue': 'N/A',
                'keywords': 'N/A',
                'cast': 'N/A',
                'crew': 'N/A',
                'original_title': 'N/A',
                'adult': 'N/A',
                'homepage': 'N/A'
            }

            # Loop through all fields and append their values
            for field, default in fields.items():
                if field == "genres":
                    genres = j.get("genres", [])
                    genres_names = "#".join([genre.get("name", "Unknown") for genre in genres]) if genres else "N/A"
                    output_line += genres_names + ";"
                elif field == "production_companies":
                    companies = j.get("production_companies", [])
                    companies_names = "#".join([company.get("name", "Unknown") for company in companies]) if companies else "N/A"
                    output_line += companies_names + ";"
                elif field == "production_countries":
                    countries = j.get("production_countries", [])
                    country_codes = "#".join([country.get("iso_3166_1", "Unknown") for country in countries]) if countries else "N/A"
                    output_line += country_codes + ";"
                elif field == "spoken_languages":
                    languages = j.get("spoken_languages", [])
                    languages_names = "#".join([language.get("name", "Unknown") for language in languages]) if languages else "N/A"
                    output_line += languages_names + ";"
                elif field == "cast":
                    # Cast data can be more complex and may require a separate API call
                    cast_data = j.get("credits", {}).get("cast", [])
                    cast_names = "#".join([actor.get("name", "Unknown") for actor in cast_data]) if cast_data else "N/A"
                    output_line += cast_names + ";"
                elif field == "crew":
                    # Crew data can be more complex and may require a separate API call
                    crew_data = j.get("credits", {}).get("crew", [])
                    crew_names = "#".join([crew_member.get("name", "Unknown") for crew_member in crew_data]) if crew_data else "N/A"
                    output_line += crew_names + ";"
                else:
                    # For other fields, just extract the data directly
                    output_line += str(j.get(field, default)) + ";"
            
            # Write the full data for the movie to the output file
            write(output_line + "\n")
        
        else:
            # Print error message if request fails
            print(f"Failed to retrieve data for movie ID {number}, Status Code: {response.status_code}")
