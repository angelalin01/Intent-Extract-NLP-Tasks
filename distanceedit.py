import re

class DistanceEdit():

    def longest_substring(self,X, Y):
        m = len(X)
        n = len(Y)
        LCSuff = [[0 for x in range(n+1)] for l in range(m+1)]
        result = 0
        for i in range(m+1):
            for j in range(n+1):
                if(i==0 or j==0):
                    LCSuff[i][j] = 0
                elif(X[i-1] == Y[j-1]):
                    LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                    result = max(result, LCSuff[i][j])
                else:
                    LCSuff[i][j]=0
        return result
    
    def iterative_levenshtein(self,s,t):
        """
        computes levenshtein distance 
        """
        rows = len(s)+1
        cols = len(t)+1
        dist = [[0 for x in range(cols)] for x in range(rows)]
        for i in range(1, rows):
            dist[i][0] = i
        for i in range(1, cols):
            dist[0][i] = i
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row-1] == t[col-1]:
                    cost = 0
                else:
                    cost=1
                dist[row][col] = min(dist[row-1][col-1]+cost, dist[row-1][col]+1, dist[row][col-1]+1)
        return dist[row][col]

    