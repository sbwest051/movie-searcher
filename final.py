import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

def get_link(movie: str) -> dict:
   """Gets the webpage links for the top three results on the rotten tomatoes search page that results from searching the movie
   and puts them in a dictionary as values where the keys are the movies that they correspond with """
   cleaned_movie = movie.replace(' ', '%20')
   url = "https://www.rottentomatoes.com/search?search={}"
   dictionary = {}
   num = 3
   try:
      resp = requests.get(url.format(cleaned_movie))
      soup = BeautifulSoup(resp.content, "html.parser").find_all('search-page-media-row')
      if len(soup) < num:
         num = len(soup)
      for result in soup:
         if len(dictionary) < num:
            title = result.find_all('img')[0].get('alt')
            year = result.get('releaseyear')
            link = result.find('a', class_="unset").get("href")
            dictionary[title + ' ' + year] = link
      return dictionary
   except:
      raise "No movie named {} found".format(movie)


def streaming_services(movie: str) -> dict:
   """Takes links of top three movies for dictionary and returns another dictionary where keys are movie titles and values 
   are the streaming services that the movie is availble on """
   output_dict = {}
   for mov, link in get_link(movie).items():
      resp = requests.get(link)
      output = []
      stream_list = BeautifulSoup(resp.content, "html.parser").find_all("where-to-watch-meta")
      for service in stream_list:
         output.append(service.get('affiliate'))
      if output == []:
         output = ['No Streaming Service Identified']
      output_dict[mov] = output
   return output_dict

def filter_services(movie: str, services: list) -> dict:
   """the main filter ser"""
   filtered = {}
   full_dict = streaming_services(movie)
   for mov, info in full_dict.items():
      serv_list = []
      for serv in info:
         if serv in services:
            serv_list.append(serv)
            filtered[mov] = serv_list
      if serv_list == []:
         filtered[mov] = ['No Streaming Service Identified']
   return filtered

from dataclasses import dataclass
import csv
import math
import re

@dataclass
class Movie:
   id: int
   title: str
   keywords: list
   popularity: float

bad_characters = re.compile(r"[^\w]")

def clean_keywords(keywords: str) -> list:
   """input: string representing the description of a movie
   output: a list with each of the words for a description
   description: this function parses through all of the words for a description and makes sure
   they contain valid characters
   """
   output = []
   cleaned = keywords.replace('[','').replace('{','').replace('"','').replace('id','').replace(
      '1','').replace('2','').replace('3','').replace('4','').replace('5','').replace(
         '6','').replace('7','').replace('8','').replace('9','').replace('0','').replace(
            'name','').replace(':','').replace('}', '').replace("'",'').replace(',','').replace(
               ']','').lower()
   final = cleaned.split(' ')
   for word in final:
      if (word != '') and (word != '\n'):
         output.append(word)
   return output



def create_corpus(filename: str) -> list:
   """input: a filename
   output: a list of Songs
   description: this function is responsible for creating the collection of songs, including some data cleaning
   """
   with open(filename) as f:
    corpus = []
    count = 0
    for s in csv.reader(f):
        if count <= 19000:
           if s[0] != 'adult':
               count += 1
               new_movie = Movie(s[5], s[20], clean_keywords(s[3] + ' ' + s[3] + ' ' + s[3] + ' ' + s[3] + ' ' + s[3] + ' ' + s[9] + ' ' + s[19]), float(s[10]))
               corpus.append(new_movie)
   return corpus

def compute_idf(corpus: list) -> dict:
   """input: a list of Songs
   output: a dictionary from words to inverse document frequencies (as floats)
   description: this function is responsible for calculating inverse document
   frequencies of every word in the corpus
   """
   appearances = {}
   idf_dictionary = {}
   for movie in corpus:
      for word in set(movie.keywords):
         if word not in appearances:
               appearances[word] = 1
         else: appearances[word] += 1
   for word, count in appearances.items():
      idf_dictionary[word] = math.log(len(corpus)/count)
   return idf_dictionary

def compute_tf(keywords: list) -> dict:
   """input: list representing the description
   output: dictionary containing the term frequency for that description
   description: this function calculates the term frequency for a set of descriptions
   """
   tf_dictionary = {}
   for word in keywords:
      if word not in tf_dictionary:
         tf_dictionary[word] = 1
      else:
         tf_dictionary[word] += 1
   return tf_dictionary


def compute_tf_idf(keywords: list, corpus_idf: dict) -> dict:
   """input: a list representing the description and an inverse document frequency dictionary
   output: a dictionary with tf-idf weights for the song (words to weights)
   description: this function calculates the tf-idf weights for a song
   """
   tf_idf_dictionary = {}
   tf_dictionary = compute_tf(keywords)
   for word in tf_dictionary:
      tf_idf_dictionary[word] = tf_dictionary[word] * corpus_idf.get(word, 0)
   return tf_idf_dictionary


def compute_corpus_tf_idf(corpus: list, corpus_idf: dict) -> dict:
   """input: a list of songs and an idf dictionary
   output: a dictionary from song ids to tf-idf dictionaries
   description: calculates tf-idf weights for an entire corpus
   """
   tf_idf_corpus = {}
   for movie in corpus:
      tf_idf_corpus[movie.id] = compute_tf_idf(movie.keywords, corpus_idf)
   return tf_idf_corpus

def cosine_similarity(l1: dict, l2: dict) -> float:
   """input: dictionary containing the term frequency - inverse document frequency weights (tf-idf) for a song,
   dictionary containing the term frequency - inverse document frequency weights (tf-idf) for a song
   output: float representing the similarity between the values of the two dictionaries
   description: this function finds the similarity score between two dictionaries by representing them as vectors and comparing their proximity.
   """
   magnitude1 = math.sqrt(sum(w * w for w in l1.values()))
   magnitude2 = math.sqrt(sum(w * w for w in l2.values()))
   dot = sum(l1[w] * l2.get(w, 0) for w in l1)
   if (magnitude1 * magnitude2) == 0: return 0
   return dot / (magnitude1 * magnitude2)

def nearest_neighbor(
   keywords: list, corpus: list, corpus_tf_idf: dict, corpus_idf: dict, movie_title: str, input_mov) -> Movie:
   """input: a string representing the description for a movie, a list of songs,
   tf-idf weights for every song in the corpus, and idf weights for every word in the corpus
   output: a Song object
   description: this function produces the song in the corpus that is most similar to the lyrics it is given
   """
   current_best_match = Movie('','No Recommendation Found','','')
   current_best_similarity = 0
   for movie in corpus:
      if movie.popularity > 10:
         if (movie.title.lower() != movie_title.lower()) and (movie.title.lower() != input_mov.lower()):
            similarity = cosine_similarity(corpus_tf_idf[movie.id], 
            compute_tf_idf(keywords, corpus_idf))
            if similarity > current_best_similarity:
               current_best_similarity = similarity
               current_best_match = movie
   return current_best_match

def main(filename: str, keywords: str, movie_title: str, input_mov):
   corpus = create_corpus(filename)
   corpus_idf = compute_idf(corpus)
   corpus_tf_idf = compute_corpus_tf_idf(corpus, corpus_idf)
   return nearest_neighbor(clean_keywords(keywords), corpus, corpus_tf_idf, corpus_idf, movie_title, input_mov).title


def movie_suggestion(keyword_list: list, movie_title: str, input_mov) -> str:
   main_input = keyword_list[0] + ' ' + keyword_list[1] + ' ' + keyword_list[1] + ' ' + keyword_list[1]
   return main('movies.csv', main_input, movie_title, input_mov)



def get_description(movie: str) -> dict:
  output_dict = {}
  for mov, link in get_link(movie).items():
     output_list = []
     resp = requests.get(link)
     overview = BeautifulSoup(resp.content, "html.parser").find(id="movieSynopsis").text.strip()
     output_list.append(overview)
     if BeautifulSoup(resp.content, "html.parser").find('div', class_="meta-value genre") != None:
        genre = BeautifulSoup(resp.content, "html.parser").find('div', class_="meta-value genre").text.strip()
        output_list.append(genre)
     output_list.append('')
     output_dict[mov] = output_list
  return output_dict

 
def give_suggestion(movie: str) -> dict:

  suggestion_dict = {}
  descriptions = get_description(movie)
  for film, descr in descriptions.items():
     suggestion = movie_suggestion(descr, film, movie)
     suggestion_dict[film] = suggestion
  return suggestion_dict

def pretty(movie:str):
   """Puts alll of the functions above together in order to make a pretty output"""
   subs = {"netflix":9.99,"disney-plus-us":10.99,"paramount-plus-us":9.99,"hbo-max":9.99}
   for mov, serv in streaming_services(movie).items():
      if serv[0]== "No Streaming Service Identified":
         print(mov + " is not available on any recognized streaming platforms." +
         "\n Based on your search we reccomend watching " + give_suggestion(movie)[mov])
         for obj in serv:
            if obj in subs:
               print("watch with subcription on "+ obj + " for " + str(subs[obj]) )
      else:
         print(mov + " is available on the following streaming platforms:" + " ".join([item for item in serv])
         + "\n Based on your search we reccomend watching " + give_suggestion(movie)[mov])
         for obj in serv:
            if obj in subs:
               print("watch with subcription on "+ obj + " for " + str(subs[obj]) )

