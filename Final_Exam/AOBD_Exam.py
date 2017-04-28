import xlwt
import xlrd
import sys
import fileinput
import json
import csv
import unicodedata

''' method to find similarity between candidates'''
def candSimilarity(candidate1, candidate2):
    similarity = []
    for skill in candidate1:
		if skill in candidate2:
			similarity.append(candidate1[skill]) 
    return similarity

''' method to find closest candidates based on skills/ career goal of the candidate '''
def closestCandidates(candidate1,candidates):
	similarity = []
	for candidate in candidates:
		if candidate != candidates:
			similarity = candSimilarity(candidate1,candidate)
			similarity.append(similarity)
	similarity.sort()
	return similarity
    
''' method to suggest career path based on skills of the candidate '''
def suggestCareerPathSkillset(candidate, candidates):
	closest = closestCandidates(candidate, candidates)
	suggestions = []
	closeCandPath = candidates
	candidateJob = candidate
	for jobTitle in closeCandPath:
		if not jobTitle in candidate:
			suggestions.append(jobTitle)
	return sorted(suggestions)

''' method to suggest career path based on career goal of the candidate '''
def suggestCareerPathGoal(candidate, candidates):
	closest = closestCandidates(candidate, candidates)
	suggestions = []
	closeCandPath = candidates
	candidateJob = candidate
	for goal in closeCandPath:
		if not goal in candidate:
			suggestions.append(goal)
	return sorted(suggestions)

''' to clean the data from file '''
book = xlwt.Workbook()
ws = book.add_sheet('First Sheet')  # Add a sheet

''' enter file name whose data is to be cleaned '''
with open('test1.txt', 'r') as file :
  filedata = file.read()

''' Replace the target string '''
filedata = filedata.replace('\u2013', '-')
filedata = filedata.replace('\uf0a7', '')
filedata = filedata.replace('\u0219', 's')
filedata = filedata.replace('\u021b', 't')
filedata = filedata.replace('\u2018', "'")
filedata = filedata.replace('\xe9', 'e')
filedata = filedata.replace('\xa3', '')
filedata = filedata.replace('\u2019', "'")
filedata = filedata.replace('\u00e9', 'e')
filedata = filedata.replace('\u0303', '~')
filedata = filedata.replace('\u00f1', 'n')
filedata = filedata.replace('\u2026', '...')
filedata = filedata.replace('\u00f6', 'o')


''' Write the file out again '''
with open('file.txt', 'w') as file:
  file.write(filedata)

''' open file for writing csv using json ''' 
with open("file.txt", "rb") as fin:
    content = json.load(fin)
with open("stringJson.txt", "wb") as fout:
    json.dump(content, fout, indent=1)
f = open('stringJson.txt', 'r+')

'''read all lines at once'''
data = f.readlines() 
with open("stringJson.txt", "rb") as fin:
	x = json.load(fin)

f = csv.writer(open("test.csv", "wb+"))

''' Write CSV Header '''
f.writerow(["CandidateID", "Company", "Job-Description", "Job Title", "Job-Duration",  "Skills", "Additional-Info", "Location", "Institute", "School-Duration", "Qualification", "Resume-Summary"])

jobTitleCand = []
jobTitleOtherCand = []
''' Take input from user'''
CandidateID_input = input("Enter your CandiateID: ")

'''Convert the given .txt file to .csv '''
for i in x:
     f.writerow([i["CandidateID"],
                 i["Work-Experience"]["Company"],
                 i["Work-Experience"]["Job-Description"],
                 i["Work-Experience"]["Job Title"],
                 i["Work-Experience"]["Job-Duration"],
                 i["Skills"],
                 i["Additional-Info"],
                 i["Location"],
                 i["Education"]["Institute"],
                 i["Education"]["School-Duration"],
                 i["Education"]["Qualification"],
                 i["Resume-Summary"]])
     ''' Decode data from unicode to ASCII for processing it. '''
     unicodedata.normalize('NFKD', i["CandidateID"]).encode('ascii','ignore')
     unicodedata.normalize('NFKD', i["Work-Experience"]["Company"]).encode('ascii','ignore')
     unicodedata.normalize('NFKD', i["Work-Experience"]["Job-Description"]).encode('ascii','ignore')
     unicodedata.normalize('NFKD', i["Work-Experience"]["Job Title"],).encode('ascii','ignore')
     unicodedata.normalize('NFKD', i["Work-Experience"]["Job-Duration"]).encode('ascii','ignore')
     unicodedata.normalize('NFKD', i["Skills"]).encode('ascii','ignore')
     unicodedata.normalize('NFKD', i["Additional-Info"]).encode('ascii','ignore')
     unicodedata.normalize('NFKD', i["Resume-Summary"]).encode('ascii','ignore')

     ''' Store job titles of each candidate'''
     if(int(i["CandidateID"]) == CandidateID_input):
    	z = unicodedata.normalize('NFKD', i["Work-Experience"]["Job Title"],).encode('ascii','ignore')
    	jobTitleCand.append(z.split(" && "))
    	
    else:
    	z = unicodedata.normalize('NFKD', i["Work-Experience"]["Job Title"],).encode('ascii','ignore')
    	jobTitleOtherCand.append(z.split(" && "))
    	
'''Compare job titles for suggesting career path using skillset'''
pathSkillSet = suggestCareerPathSkillset(jobTitleCand, jobTitleOtherCand)
pathGoal = suggestCareerPathSkillset(jobTitleCand, jobTitleOtherCand)

print final
print goal

''' This will return a line of string data, you may need to convert to .xls '''
#for i in range(len(data)):
#	row = data[i].split('", "')  
#	for j in range(len(row)):
#		ws.write(i, j, row[j])  # Write to cell i, j
#book.save('test' + '.xls')
# f.close()
