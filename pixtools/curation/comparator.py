import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from pixels import Experiment
#Before using any function please read its warnings, as they may affect your usage and the validity of their output


def id_merges(path, df1):
#This function IDs all merges done in a session. It is not thought to use directly, so you will have to introduce the path to the clusters_info.tsv if you want it to work outside other functions. If the way phy represents merges in its logs or id-wise, this programme will become unreliable.
	regex="(Merge clusters )(.*)(, )(.*)( to )(.*)(\.)" #Regular expression to identify the merges in the phy.log
	path3=Path(path).parent
	path3=Path(f"{path3}/phy.log") #This selects the proper path to the log
	idlist=list(df1.id)
	merged_ids=list()
	for i in range(0, idlist[-1]): #This loop generates a list with all missing ids in the .tsv, which are the ones that were merged.
		if i not in idlist:
			merged_ids.append(i)
	matchlist=list()
	merges=dict()
	ablist=list()
	clist=list()
	raw_clusters_m=list()#All of these lists and dictionaries do various things afterwards, like keeping tabs on the detected merges
	with open(path3, "r") as file:
		for line in file:
			for match in re.finditer(regex, line, re.S): #This loop does the matching between the regex and each line
				match=match.group() #This is necessary to be able to manipulate the string of interest
				matchlist.append(match) #This list is necessary to be able to revert it
		for matches in reversed(matchlist):#This loop goes through the matches in inverse order to overcome the "undone" problem
			pmerge=list(re.findall(r'\b\d+\b', matches))#This will extract all numbers from strings separated by . ,
			a=int(pmerge[0])
			b=int(pmerge[1])
			c=int(pmerge[2])
			if a in merged_ids and b in merged_ids: #This will check that the clusters are actually missing in the .tsv
				if c not in clist and a not in ablist and b not in ablist: 
#This if statement is necessary to solve the "merge, undone, merge" problem thanks to the reversed loop. 
					exec('''merges["%d + %d"]=c''' % (a, b))
					ablist.append(a)
					ablist.append(b)
					clist.append(c)
					raw_clusters_m.append(a)
					raw_clusters_m.append(b)
					raw_clusters_m.append(c)
	file.close()#I do not know if this is necessary but it does no harm
	if cM==False:
		exec('''print("""List of merges from %s:""")''' % (path))
		[print(key,':',value) for key, value in merges.items()]
		return raw_clusters_m
	if cM==True:
		return merges


def compare_curation(experiment, excludeNoise=False):
#Calling this function will provide you with a number of graphs comparing the number of labels for each type of label. Each graph will comprise all curations from an experimental session and the KiloSort labels, and there will be as many graphs as experimental sessions there are for the selected mice. experiment value refers to the Experiment object, and you will need ot make sure the directories are correct as well as the mouse ID. 
#WARNING1(outdated, merges accounted for):Only eliminate noise when you are comparing with KSLabel and only if you are willing to accept a margin of error, as eliminating the noise labels is not accurate because you could be eliminating good clusters from another session in the process. This happens because merges are unaccounted for, so the possible error should only be taken when comparing with KSLabel, if you are doing it you probably want to use the next two functions for that
#WARNING2: Make sure the curation sessions are all in different folders inside the session folder, always following a sorted_0* format for the name to recognise each session (The name requirement could change in future updates). If not, the curation sessions that do not follow this indication will be considered as isolated experimental sessions and not compared to the rest.
#WARNING3: There are a lot of indentations, so it may look like the script has jumped to the next line wihtout an indentation when in fact the number of indentations makes the line jump, and everything works fine. To avoid this effect read this file in a bigger window/tab, to really notice the indentations.
#WARNING4(outdated, merges accounted for): The results of this analysis is qualitative until merges can be accounted for, and other things are also fixed. Do not use as a quantitative analysis
	for ids in experiment.mouse_ids:#Separates mice put on the Experiment object
		print(ids)
		pathlist =(list(path for path in glob.glob(f"{experiment.data_dir}/processed/*{ids}/**/cluster_info.tsv"))) #Makes a list of each directory with a cluster_info.tsv file for each mouse
		count1=0 #Only updates in each loop, generates new df name
		while pathlist:
			legend=dict() #This will generate a legend for each graph
			count2=0 #Updates after each dataset added, names each column
			exec('groupdata%d=pd.DataFrame()' % (count1)) #This generates a dataframe with name for each session depending on the loop, to avoid needing an outside input.
			path=pathlist.pop(0)#Ensures the while loop will end and checks every path is dealt with
			df1=pd.read_table(f"{path}", sep="	")#.tsv is read
			merges=id_merges(path, df1)
			df2 = df1[~df1.id.isin(merges)]#Merges deleted and printed apart
			exec('groupdata%d["id"]=df2["id"]' % (count1))#Adds the IDs to the dataframe for merge purposes. Deleted before graphic.
			exec('groupdata%d["KSLabel"]=df2["KSLabel"]' % (count1))
			exec('groupdata%d[1]=df2["group"]' % (count1)) #Up until here there is a loop-generated dataframe, a .tsv dataframe with the merges deleted and printed apart in a dictionary, and the loop-generated dataframe has been filled with the KSLabel column and the human-curated one.
			legend["1"]=path #This is added to a dictionary as a legend
			count2 += 1
			parentpath= Path(path).parents[1] #Necessary for comparison later
			index = 0
			while index < len(pathlist):
				parentpaths= Path(pathlist[index]).parents[1] 
				if parentpath == parentpaths: #Compares grandparent directories (-2 from file) to aggregate in sessions, reason for WARNING2
					count2 += 1
					paths=pathlist.pop(index)
					df3=pd.read_table(f"{paths}", sep="	")
					merges.append(id_merges(paths, df3))
					df4=df3[~df3.id.isin(merges)] #Merge stuff that is necessary, ask Toni
					exec('groupdata%d=groupdata%d[~groupdata%d.id.isin(merges)]' % (count1, count1, count1))
					exec('groupdata%d[%d]=df4["group"]' % (count1, count2))  #Same as before, bye merges hi to the previous loop-generated dataframe
					legend[str(count2)]=paths
				else:
					index += 1
			count1 += 1 #Number that generates the new dataframe names
#Now we have created a number of groupdata* DataFrames that each one contains the KSLabel, and the curated_label for each curation and there is one dataframe for each session.
			print("Legend of table number ", count1)
			[print(key,':',value) for key, value in legend.items()]
		if excludeNoise == True: #This will delete rows with "noise" labels
			for dataf in range (0, count1):
				exec("""groupdata%d = groupdata%d[~groupdata%d.eq("noise").any(1)]""" % (dataf, dataf, dataf))
		sns.set_theme(style="whitegrid") #Could be changed depending on your preference
		for count in range(0, count1):
			exec('groupdata%d=groupdata%d.drop(columns=["id"], axis=1)' % (count, count)) #IDs deleted for graph purposes
			exec('groupdata%d=groupdata%d.reset_index().melt("index", var_name="sessions", value_name="Label")' % (count, count)) #This converts the dataframe to one that can be used by seaborn, changing the column names to a row that contains the preious column information
			exec('groupdata%d["sessions"]=groupdata%d["sessions"].astype("category")' % (count, count))	#This will convert the data in column "sessions" into categorical data for the graph
			#exec('print (groupdata%d)' %(count)) #This is for troubleshooting purposes
			exec('ax%d = sns.countplot(x="sessions", hue="Label", data=groupdata%d)' % (count, count))
			plt.show() #Both these lines create and print the graphs.


def different_clusters(experiment, excludeNoise=False, printAll=True):
#This function will extract all cluster labels from all sessions from all mice and from all curation sessions properly stored (see WARNING2) and will put them in a common dataframe taking into account merges and with ID equity. Then it will compare ID by ID and group all clusters depending on what change there is between it and the one it is being compared to. Previously to those lists there will be a list where all different clusters' ids are printed telling you (yes, you) what sessions are compared with single numbers. If you want to know what sessions are they you will have to scroll upwards to a "Legend" entry, where it tells you which path is which number (I know, what a pain).
	for ids in experiment.mouse_ids:#Separates mice put on the Experiment object
		print(ids)
		pathlist =(list(path for path in glob.glob(f"{experiment.data_dir}/processed/*{ids}/**/cluster_info.tsv"))) #Makes a list of each directory with a cluster_info.tsv file for each mouse
		count1=0 #Only updates in each loop, generates new df name
		while pathlist:
			legend=dict() #This will generate a legend for each graph
			count2=0 #Updates after each dataset added, names each column
			exec('groupdata%d=pd.DataFrame()' % (count1)) #This generates a dataframe with name for each session depending on the loop, to avoid needing an outside input.
			path=pathlist.pop(0)#Ensures the while loop will end and checks every path is dealt with
			df1=pd.read_table(f"{path}", sep="	")#.tsv is read
			merges=id_merges(path, df1)
			df2 = df1[~df1.id.isin(merges)]#Merges deleted and printed apart
			exec('groupdata%d["id"]=df2["id"]' % (count1))#Adds the IDs to the dataframe for merge purposes. Deleted before graphic.
			exec('groupdata%d["KSLabel"]=df2["KSLabel"]' % (count1))
			exec('groupdata%d[1]=df2["group"]' % (count1)) #Up until here there is a loop-generated dataframe, a .tsv dataframe with the merges deleted and printed apart in a dictionary, and the loop-generated dataframe has been filled with the KSLabel column and the human-curated one.
			legend["1"]=path #This is added to a dictionary as a legend
			count2 += 1
			parentpath= Path(path).parents[1] #Necessary for comparison later
			index = 0
			while index < len(pathlist):
				parentpaths= Path(pathlist[index]).parents[1] 
				if parentpath == parentpaths: #Compares grandparent directories (-2 from file) to aggregate in sessions, reason for WARNING2
					count2 += 1
					paths=pathlist.pop(index)
					df3=pd.read_table(f"{paths}", sep="	")
					merges.append(id_merges(paths, df3))
					df4=df3[~df3.id.isin(merges)] #Merge stuff that is necessary, ask Toni
					exec('groupdata%d=groupdata%d[~groupdata%d.id.isin(merges)]' % (count1, count1, count1))
					exec('groupdata%d[%d]=df4["group"]' % (count1, count2))  #Same as before, bye merges hi to the previous loop-generated dataframe
					legend[str(count2)]=paths
				else:
					index += 1
#Now we have created a number of groupdata* DataFrames that each one contains the KSLabel, and the curated_label for each curation and there is one dataframe for each session.
			print("Legend of comparison number ", count1+1)
			[print(str(key),':',value) for key, value in legend.items()] #Prints legend
			if excludeNoise == True: #Eliminates noise
				exec("""groupdata%d = groupdata%d[~groupdata%d.eq("noise").any(1)]""" % (count1, count1, count1))
			if count2<2:
				continue
			for count in range(1, count2): #All of the following does the comparison
				if printAll==True:
					exec('print("All differences in %d vs %d", list(groupdata%d[groupdata%d[%d]!=groupdata%d[%d]].id))' % (count, count+1, count1, count1, count, count1, count+1))
				exec('print("Good to mua in %d vs %d", list(groupdata%d[(groupdata%d[%d]=="good") & (groupdata%d[%d]=="mua")].id))' % (count, count+1 ,count1, count1, count, count1, count+1))
				if excludeNoise==False:
					exec('print("Good to noise in %d vs %d", list(groupdata%d[(groupdata%d[%d]=="good") & (groupdata%d[%d]=="noise")].id))' % (count, count+1 ,count1, count1, count, count1, count+1))
					exec('print("Noise to good in %d vs %d", list(groupdata%d[(groupdata%d[%d]=="noise") & (groupdata%d[%d]=="good")].id))' % (count, count+1 ,count1, count1, count, count1, count+1))
				exec('print("Mua to good in %d vs %d", list(groupdata%d[(groupdata%d[%d]=="mua") & (groupdata%d[%d]=="good")].id))' % (count, count+1, count1, count1, count, count1, count+1))
				if excludeNoise==False:
					exec('print("Mua to noise in %d vs %d", list(groupdata%d[(groupdata%d[%d]=="mua") & (groupdata%d[%d]=="noise")].id))' % (count, count+1 ,count1, count1, count, count1, count+1))
					exec('print("Noise to mua in %d vs %d", list(groupdata%d[(groupdata%d[%d]=="noise") & (groupdata%d[%d]=="mua")].id))' % (count, count+1 ,count1, count1, count, count1, count+1))
			count1 += 1 #Number that generates the new dataframe names

def KS_comparation_idlist(experiment, excludeNoise=True):
#This function will print all mismatched IDs in labelling between KiloSort and human curation in a list. It will exclude noise labels
	for ids in experiment.mouse_ids:
		print(ids)
		pathlist =(list(path for path in glob.glob(f"{experiment.data_dir}/processed/*{ids}/**/cluster_info.tsv"))) #Makes a list of each directory with as cluster_info.tsv for each mouse
		while pathlist:
			path=pathlist.pop(0) #This ensures the while loop terminates
			print("Comparing ", path)
			dfcomp=pd.read_table(f"{path}", sep="	")
			if excludeNoise ==True:
				dfcomp = dfcomp[~dfcomp.eq("noise").any(1)] #This line eliminates the "noise" labels
			print("All ", list(dfcomp[dfcomp["KSLabel"]!=dfcomp["group"]].id)) #Prints list of ids
			print("Good to mua ", list(dfcomp[(dfcomp.KSLabel=="good") & (dfcomp.group=="mua")].id))
			if excludeNoise==False:
				print("Good to noise ", list(dfcomp[(dfcomp.KSLabel=="good") & (dfcomp.group=="noise")].id))
			print("Mua to good ", list(dfcomp[(dfcomp.KSLabel=="mua") & (dfcomp.group=="good")].id))
			if excludeNoise==False:
				print("Mua to noise ", list(dfcomp[(dfcomp.KSLabel=="mua") & (dfcomp.group=="noise")].id))
