'''
  File name: code_snippets.py
  Author: 
  Date created: 11/4/2018

  Use this for code snippets you need to cut but don't want to lose
'''

	# Calculate the jacobians using 11 x 11 window so I can do convolution to create sums
	xkernel = np.ones(11).reshape(1, 11)
	ykernel = np.ones(11).reshape(11, 1)