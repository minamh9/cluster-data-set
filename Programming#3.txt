
# Importing Libraries
import numpy as np # for working with arrays
import pandas as pd  # for data analysis
import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For data visualization
import random # For random data point generation

# Loading the dataset into a pandas dataframe
df = pd.read_csv("/content/545_cluster_dataset programming 3.txt",header = None,sep="  ")

# Dataset visualization
sns.scatterplot(df.values[:,0],df.values[:,1])
plt.show()

# Euclidean distance of two 2D data points
def dist(x1,x2):
  '''
  This function takes two 2d data points and computes the euclidean distance between them
  '''
  return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)

def assignment(k,X,centroids):
  '''
  This function is the assigment step of kmeans algorithm
  Inputs: 
  k: number of clusters
  X: The data points
  centroids: The initial or former step's centroids
  Output:
  cluster_idx : The cluster id assigned to data points
  '''
  # initializing a matrix of zeros for distances of each data point to each centroid
  # with the shape of (number of data points, number of clusters)
  dist_to_centers = np.zeros((X.shape[0],k))
  # Looping over data points
  for j in range(X.shape[0]):
    # Looping over number of clusters
    for i in range(k):
      # Using the dist function two compute the euclidean distance between each data point and centroid
      dist_to_centers[j,i] = dist(X[j],centroids[i])
  # Finding the cluster idx by taking argmin of the distances 
  cluster_idx = np.argmin(dist_to_centers,axis=1)
  return cluster_idx

def update(k, cluster_idx,X):
  '''
  This function updates the centriods 

  Inputs:
  k : number of clusters
  cluster_idx : cluster indexes
  X : data points
  Outputs:
  centroids: newly computed centroids 
  '''
  # Initializing centroids with an empty python list
  centroids = []

  # Looping over number of clusters
  for i in range(k):
    # taking the mean of data points which are from the same cluster and appending it to the list
    centroids.append(np.mean(X[cluster_idx==i],axis=0))
  return centroids

def distances_to_centroids(input, clusters, centroids):
  '''
  This function computes the Sum of squared distances of data points to their closest cluster center (wcss)
  Inputs: 
  input: data points
  clusters: final clusters
  centroids: final centroids
  Outputs:
  sum square error of distances (wcss)
  '''
  # setting the sum to zero
  sum_distances = 0
  # Finding number of clusters
  num_clusters = max(clusters) +1 
  # Looping over number of clusters
  for k in range(num_clusters):
    # Selecting the data related to each cluster
    data = input[clusters == k]
    # Selecting the centeroid related to each cluster
    center = centroids[k]
    # Looping over the data points of each cluster
    for j in range(data.shape[0]):
      # Adding square of distances to the final wcss
      sum_distances += dist(data[j],center)**2
  return sum_distances

def make_initial_centriods(input, k):
  '''
  A function to generate initial centroids
  Inputs:
  input: data points
  k = number of clusters
  Outputs: 
  initial centroids : the initial chosen centroids from the input data points
  '''
  # Random ndex of centroids with size of k
  cent_idx = random.sample(range(input.shape[0]),k)
  initial_centroids = []
  # Appending the selected initial centroids to the list
  for i in cent_idx:
      initial_centroids.append(input[i])
  return initial_centroids

def k_means(input,k, r =10):
  '''
  Main kmeans function that perform assigment and update steps, as well as r different inits
  and computes the wcss errors and choose the best one
  Inputs:
  input : data points
  k : number of clusters
  r : number of inits
  Outputs:
  all_iter_clusters[best_iter][-1]: The best cluster indexes from all the inits
  all_iter_centriods[best_iter][-1]: The best cluster centroids from all the inits
  errors[best_iter]: The lowest wcss error of best init
  best_iter:  The index of best iteration (r)
  all_iter_clusters: All the clusters from all the inits and from inside each init to help with plots
  all_iter_centriods : All the centroids from all the inits and from inside each init to help with plots
  '''

  # All the clusters and centroids from each init and from each step
  all_iter_clusters = []
  all_iter_centriods = []
  errors = []
  # Looping over number of inits (r)
  for iter in range(r):
    print (f'----------------- r = {iter}----------------')

    # Making initial centroids
    init_centroids = make_initial_centriods(input, k)

    # all clusters inside each init at each step
    all_clusters = []
    initial_clusters = np.zeros((input.shape[0],))
    all_clusters.append(initial_clusters)
    # Loss is the difference of new cluster indexes and the previous one
    # If zero means there is no change to cluster indexes and we stop it
    loss = 1
    i = 0
    all_centroids = []
    all_centroids.append(init_centroids)

    # Continue to do assigment and update steps until the loss converges to zero
    # If zero means there is no change to cluster indexes and we stop it
    while loss != 0: 

      # Assignment step
      clusters = assignment(k,input,all_centroids[i])
      # Update step
      new_centroids = update(k, clusters,input)
      all_clusters.append(clusters)
      all_centroids.append(new_centroids)
      # Loss of each iteration, difference between cluster indexes and previous step cluster indexes
      loss = np.sum(clusters != all_clusters[i])
      i+=1
      print('iteration: ', i, 'loss: ', loss)

    # WCSS error at the end of each init (r)
    error = distances_to_centroids(input, all_clusters[-1], all_centroids[-1])
    errors.append(error)

    print('error', error)

    all_iter_clusters.append(all_clusters)
    all_iter_centriods.append(all_centroids)

  # Best init error index
  best_iter = np.argmin(errors)
  print('--------------')
  print(f'best r is {best_iter} with sum square error of {errors[best_iter]}')
  print('--------------')
  return all_iter_clusters[best_iter][-1],all_iter_centriods[best_iter][-1],errors[best_iter],best_iter,  all_iter_clusters, all_iter_centriods

# Running our kmeans for k =3 on our dataset with 10 different random inits (r)
X = df.values
best_cluster, best_centroid, best_error,best_iter, clusters, centroids = k_means(X, k=3, r= 10)

# Visualizing the effect of different inits (r)
for r_select in range(10):
  plt.figure()
  sns.scatterplot(x=df.values[:,0],y=df.values[:,1],hue = clusters[r_select][-1])
  plt.scatter(x=centroids[r_select][-1][0][0],y=centroids[r_select][-1][0][1],c = "red")
  plt.scatter(x=centroids[r_select][-1][1][0],y=centroids[r_select][-1][1][1],c = "blue")
  plt.scatter(x=centroids[r_select][-1][2][0],y=centroids[r_select][-1][2][1],c = "yellow")
  plt.title(f'r = {r_select}')
  plt.show()

# Visualizing each step in converging for the best selected init
number_iter = len(centroids[best_iter])
for iter in range(number_iter):
  plt.figure()
  sns.scatterplot(x=df.values[:,0],y=df.values[:,1],hue = clusters[best_iter][iter])
  plt.scatter(x=centroids[best_iter][iter][0][0],y=centroids[best_iter][iter][0][1],c = "red")
  plt.scatter(x=centroids[best_iter][iter][1][0],y=centroids[best_iter][iter][1][1],c = "blue")
  plt.scatter(x=centroids[best_iter][iter][2][0],y=centroids[best_iter][iter][2][1],c = "yellow")
  plt.title(f' iteration = {iter}')
  plt.show()

# Visualizing the effect of number of clusters (k)
for k_choice in [2, 3, 4 ,5]:
  best_cluster, best_centroid, best_error, _,_, _ = k_means(X, k=k_choice, r= 10)
  plt.figure()
  sns.scatterplot(x=df.values[:,0],y=df.values[:,1],hue = best_cluster)
  for i in range(k_choice):
    plt.scatter(x= best_centroid[i][0],y=best_centroid[i][1],c = 'red')
  plt.title(f'k = {k_choice}')
  plt.show()

