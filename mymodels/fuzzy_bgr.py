import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os 

def preparing_rgb_values(bl,gr,r):
	if  bl >= 40 :
		if gr > r+4:
			if bl >= 55:
				bl = bl+30
				r = r-25
	return bl, gr, r

def fuzzy_ripe_index(blue, green, red, im_path):

	Blue = ctrl.Antecedent(np.arange(0, 100.5, 0.5), 'Blue')
	Green = ctrl.Antecedent(np.arange(0, 121, 1), 'Green')
	Red = ctrl.Antecedent(np.arange(0, 141, 1), 'Red')

	Index = ctrl.Consequent(np.arange(0, 10.5, 0.5), 'Index')
	# Blue-membership function
	Blue['L_blue'] = fuzz.trapmf(Blue.universe, [0, 0, 35, 46])
	Blue['M_blue'] = fuzz.trapmf(Blue.universe, [45, 47, 47, 50])*0.7
	Blue['H_blue'] = fuzz.trapmf(Blue.universe, [49, 59, 69, 79])
	Blue['vH_blue'] = fuzz.trapmf(Blue.universe, [79, 80, 100, 100])

	# Green-membership function
	Green['L_green'] = fuzz.trapmf(Green.universe, [0, 0, 70, 90])
	#Green['M_green'] = fuzz.trimf(Green.universe, [161, 180, 200])
	Green['H_green'] = fuzz.trapmf(Green.universe, [89, 109, 120, 120])

	# Red-membership function
	Red['vL_red'] = fuzz.trapmf(Red.universe, [0, 0, 40, 80])
	Red['L_red'] = fuzz.trapmf(Red.universe, [80, 80, 110, 110])
	#Red['M_red'] = fuzz.trimf(Red.universe, [94, 114, 124])
	Red['H_red'] = fuzz.trapmf(Red.universe, [109, 130, 140, 140])


	# Index membership functions
	Index['Under ripe'] = fuzz.trapmf(Index.universe, [0, 0, 1, 4])
	Index['About to ripe'] = fuzz.trimf(Index.universe, [3.5, 5, 6.5])
	Index['Ripe'] = fuzz.trapmf(Index.universe, [6, 9, 10, 10])

	# Rules
	rule1 = ctrl.Rule((Blue['H_blue'] & Green['H_green'] & Red['H_red']) | 
					(Blue['H_blue'] & Green['H_green'] & Red['L_red']) |
					(Blue['H_blue'] & Green['L_green'] & Red['L_red']) |
					(Blue['H_blue'] & Green['L_green'] & Red['H_red']), Index['About to ripe'])
	rule2 = ctrl.Rule((Blue['M_blue'] & Green['H_green'] & Red['H_red']) | 
					(Blue['M_blue'] & Green['H_green'] & Red['L_red']) |
					(Blue['M_blue'] & Green['L_green'] & Red['L_red']) |
					(Blue['M_blue'] & Green['L_green'] & Red['H_red']), Index['About to ripe'])

	rule3 = ctrl.Rule(Blue['vH_blue'] & Red['vL_red'], Index['Under ripe'])

	rule4 = ctrl.Rule((Blue['L_blue'] & Red['H_red']) |
					  (Blue['M_blue'] & Red['H_red']) , Index['Ripe'])

	indexing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
	indexing = ctrl.ControlSystemSimulation(indexing_ctrl)
	#try:
	bl, gr, re = preparing_rgb_values(blue,green,red)
	indexing.input['Blue'] = bl
	indexing.input['Green'] = gr
	indexing.input['Red'] = re
	indexing.compute()
	#ind = indexing.output['Index']

	Blue.view(sim=indexing)
	plt.savefig(os.path.join(im_path, "mymodels/output_images/",'blue_mf.png'))
	plt.close()

	Green.view(sim=indexing)
	plt.savefig(os.path.join(im_path, "mymodels/output_images/",'green_mf.png'))
	plt.close()

	Red.view(sim=indexing)
	plt.savefig(os.path.join(im_path, "mymodels/output_images/",'red_mf.png'))
	plt.close()

	Index.view(sim=indexing)
	plt.savefig(os.path.join(im_path, "mymodels/output_images/",'index.png'))
	plt.close()
	#except ValueError:
	#	print ("Please be sure that the input image is for Canary Melon and captured in the specified cameras")
	
	return indexing.output['Index']