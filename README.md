# Repository for DIA Project
Data Intelligence Application project, based on social networks and matching applications

## Goal
The goal is modeling a scenario in which a matching application exploits social influence to attract more and more users. 
## About the project
The matching application needs to learn both some information on the social networks and the probabilities that the pairs of nodes will actually match after the matching has been suggested to them.

1. Imagine:
	1. a social network composed of thousands of nodes;
	2. the nodes are characterized of some features (feel free to choose which ones) and a special feature, say f* , which can assume three different values (say, A, B, C);
	3. the edges are characterized of features, such that, for every feature of the nodes (except the special feature), there is a feature of the edges representing the similarity between the values of the nodes (to clarify: consider an edge, say e , and its nodes, say n1 and n2 , and consider a feature of the nodes, say f1 ; there is a feature of the edge, say s1 , expressing the similarity between n1 and n2 for what concerns f1 ; for example, you could define s1 ( e ) = | f1 ( n1 ) - f1 ( n2 )|, where the smaller the value the higher the similarity);
	4. the activation probabilities of the edges of the social network are linear functions in the values of the features of the edges;
	5. the nodes of the network can take part into a matching application where the graph is bipartite such that the left-hand set is composed of nodes with f* =A,B, while the right-hand set is composed of nodes with f* =C and another fixed set of users that is not in the network (call the set of these nodes D);
	6. the value of matching two nodes is given by the product of a constant and the probability that the two nodes will accept the suggested matching; both the constant and the probability depends on the pair of the values of the special feature. That is, (A,C), (A,D), (B,C), (B,D).

2. Assume that there are three messages that can be spread over the network, each corresponding to a different value of the special feature f* . Every node activates or not the neighbors independently of the message. However, a node with f* =A will take part into the matching application only if activated by message A (the same for B and C). Basically, this is equivalent to say that we have a unique social network, but we have three independent cascades that can activate, one for message A, one for message B, and one for message C. Design an algorithm maximizing the social influence in the network once a budget is given for a single message, and apply it to message A. Plot the approximation error as the parameters of the algorithms vary for every specific network.

3. Apply a combinatorial bandit algorithm to the situation in which the activation probabilities are not known and we can observe the activation of the edges. Here, the goal is to maximize the social influence for message A. Plot the cumulative regret as time increases.

4. Design a bipartite matching algorithm in which the nodes taking part into the matching are the nodes activated in the social network by message a single message, and apply to message A. Assume that the activation probabilities are known and apply a combinatorial bandit algorithm to learn the probabilities related to the matches. Plot the cumulative regret as time increases. 

5. You are given a cumulative budget for three messages A, B, and C. Design an algorithm allocating the best budget to the three messages to maximize the value of the matching when all the information is known. Suggestion: assume that the budget allocated to the three messages is discretized and the discretization is coarse ( m is the number of different values). The best allocation can be found by enumerating all the possible combinations of allocation of budget to the three messages. The algorithm has a complexity that is O( m ^2). Use this algorithm in the following.

6. Design a combinatorial bandit algorithm and apply it to the case in which all the probabilities (related to the matching problem or to the edge activation) are unknown. Plot the cumulative regret as time increases.

# For personal use of the team

#### Little introduction to github
Very basic introduction of github in command line.

Important you have to open the command line in the directory inside the repo's folder.

	#for example
	~/dia_project $ 


##### Updating our folder:
For updating our folder or spacework we need to pull all the files that are in the cloud to synchronize it, the idea is to do this every time we will do a commit to avoid troubles.

	git pull

##### Selecting files that will be uploaded the cloud of github
This is called commiting, so first of all we need to choice the files we will update in the repository with the comand git add

	git add nameOfTheFile.file
	#Or if we want to update all the files we editted
	git add --all

##### Create the commit and upload it
After selecting the files we need to create a commit (name of the update) with the following line of code, the we upload it with git push

	git commit -m "Name of the commit, try to be concise"
	#finally we push it to the repository
	git push

##### Saving our credentials
Using this helper will store your passwords unencrypted on disk, protected only by filesystem permissions but it easier to work with this.

	git config credential.helper store
	git push
		Username: <type your username>
		Password: <type your password>

	#[several days later]
	git push
	#[your credentials are used automatically]



