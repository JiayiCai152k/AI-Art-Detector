import pandas as pd
from geopy.geocoders import Nominatim
import requests

# Replace YOUR_API_KEY with your actual API key. Sign up and get an API key on https://www.geoapify.com/ 
API_KEY = "00d961300fd24bd8bbfcac84a0acf0d4"


df = pd.read_csv('food_order_updated.csv')

#Setting up Nominatim
geolocator = Nominatim(user_agent="my-application1",timeout=200)


def query_google(address, API_KEY):
   longitude, latitude = '', ''
   # Build the API URL
   url = f"https://api.geoapify.com/v1/geocode/search?text={address}&limit=1&apiKey={API_KEY}"

   # Send the API request and get the response
   response = requests.get(url)

   # Check the response status code
   if response.status_code == 200:
      # Parse the JSON data from the response
      data = response.json()

      # Extract the first result from the data
      if(len(data["features"])):
         result = data["features"][0]

         # Extract the latitude and longitude of the result
         latitude = result["geometry"]["coordinates"][1]
         longitude = result["geometry"]["coordinates"][0]

      print(f"Latitude: {latitude}, Longitude: {longitude}")
   else:
      print(f"Request failed with status code {response.status_code}")

   return latitude, longitude

address_list = []
borough_list = []
city_list = []
zipcode_list = []
country_list = []
longitude_list = []
latitude_list = []
for idx, name in enumerate(df['restaurant_name']):
   print('{} of {}'.format(idx, len(df)))
   address = geolocator.geocode(name+' NYC')
   
   if(address is not None):
      print(address)
      #print(address[0].split(','))
      address_query = address[0].split(',')
      address = address_query[0] + ' ' + address_query[1] + ' ' + address_query[2]
      address_list.append(address)
      #print('address: {}'.format(address))

      borough = address_query[-6]
      borough_list.append(borough)
      #print('borough: {}'.format(borough))
      
      city = address_query[-3]
      city_list.append(city)
      #print('city: {}'.format(city))

      zip_code = address_query[-2]
      zipcode_list.append(zip_code)
      #print('zip_code: {}'.format(zip_code))

      country = address_query[-1]

      country_list.append(country)

      #address = "1600 Amphitheatre Parkway, Mountain View, CA"
      try:
         lat, long = query_google(address, API_KEY)
         latitude_list.append(lat)
         longitude_list.append(long)
      except ValueError as err:
         print({str(err)})
         latitude_list.append('')
         longitude_list.append('')
      #print('country: {}'.format(country))
   else:
      address_list.append('')
      borough_list.append('')
      city_list.append('')
      country_list.append('')
      zipcode_list.append('')
      latitude_list.append('')
      longitude_list.append('')

print("address_list: {}".format(address_list))

df['address'] = address_list
df['borough'] = borough_list
df['city'] = city_list
df['zipcode'] = zipcode_list
df['country'] = country_list
df['longitude'] = longitude_list
df['latitude'] = latitude_list

print(df.head())
df.to_csv('food_order_updated.csv')