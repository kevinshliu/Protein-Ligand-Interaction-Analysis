#####################################################################
# Score the interactions between several ligand poses 
# and a given protein of interest. If a set of residues 
# are specified, then we will score the interactions between
# the given residues and the atoms of the ligands. 
# We will use different definitions for interaction strength, the
# most fundamental one being the number of ligand atoms
# within a given cutoff distance of the protein atoms or more
# specifically, atoms of residues of interest.
# 
# Added new scoring metrics to the output file. Will output a file with the
# the following fields.
#	Lig_name, Pose ID, Nc_R1, Nc_R2, Nc_R3, Autodock_Score, Contact_variance
#	Nc_R1, Nc_R2, Nc_R3 are the number of contacts with residues of interest,
#	in this case, R1, R2 and R3 weighted by the KFC2 confidence scores of
#	R1, R2 and R3  and their respective conservation scores.
# Author: Sambit K Mishra
# Created: 02/06/19
#####################################################################

import numpy as np
import sys
import re
import matplotlib.pyplot as plt
import os
import glob
import statistics

# parse the protein PDB file as a dictionary 
def parse_protein_as_dict(filename):
	# parse the contents of a PDB file as a dictionary. This is especially useful
	# if the PDB file has more than one chain.
	PDB_struct = {}
	fh = open(filename, 'r')
	for line in fh:
		if re.match('^ATOM', line):
			chain = line[21]
			X_coord = line[30:38]
			Y_coord = line[38:46]
			Z_coord = line[46:54]
			
			# Strip white space
			X_coord = re.sub('\s+','',X_coord)
			Y_coord = re.sub('\s+','',Y_coord)
			Z_coord = re.sub('\s+','',Z_coord)
			
			# convert from string to numeric
			X_coord = float(X_coord)
			Y_coord = float(Y_coord) 
			Z_coord = float(Z_coord)	
			
			res_name = line[17:20]
			res_id = line[22:26]
			
			res_name = re.sub('\s+','',res_name)
			res_id = re.sub('\s+','',res_id)
			res_id = int(res_id)
			atomname = line[13:16].replace(' ', '')
			uniq_res_id = chain + '_' + res_name + '' + str(res_id) + '_' + str(atomname)
			uniq_res_name = chain + '_' + res_name + str(res_id)
			if 'H' not in atomname: # Only consider information for heavy atoms. Skip records having hydrogen atoms.
				if chain not in PDB_struct.keys():
					PDB_struct[chain] = dict()
					PDB_struct[chain]['COORD'] = []
					PDB_struct[chain]['RESNAME'] = []
					PDB_struct[chain]['RESID'] = []
					PDB_struct[chain]['ATOM_NAME'] = []
					PDB_struct[chain]['UNIQ_RES_ID'] = []
					PDB_struct[chain]['UNIQ_RES_NAME'] = []
				PDB_struct[chain]['COORD'].append([X_coord, Y_coord, Z_coord])
				PDB_struct[chain]['RESNAME'].append(res_name)
				PDB_struct[chain]['RESID'].append(res_id)
				PDB_struct[chain]['ATOM_NAME'].append(atomname)
				PDB_struct[chain]['UNIQ_RES_ID'].append(uniq_res_id)
				PDB_struct[chain]['UNIQ_RES_NAME'].append(uniq_res_name)
	fh.close()
	return PDB_struct


# Parse the ligand conformations PDB file as a dictionary. There will be multiple
# poses for a given ligand (as output from AutoDock Vina).
def parse_ligand_as_dict(filename):
	lig_struct = {}
	#print("file name = ", filename)
	fh = open(filename, 'r')
	for line in fh:
		if line.startswith('MODEL'):
			model_num = line[5:len(line)].replace(' ', '')
			model_num = model_num.rstrip()
			#print ("model num = ", model_num)
		elif line.startswith('REMARK VINA RESULT'):
			model_energy = line[24:29]
		elif line.startswith('ATOM'):
			atom_id = int(line[7:11].replace(' ', ''))
			atom_name = line[13:16].replace(' ', '')
			X_coord = float(line[30:38].replace(' ', ''))
			Y_coord = float(line[38:46].replace(' ', ''))
			Z_coord = float(line[46:54].replace(' ', ''))
			if 'H' not in atom_name: # Skip records for hydrogen atoms
				if model_num not in lig_struct.keys():
					lig_struct[model_num] = dict()
					lig_struct[model_num]['COORD'] = []
					lig_struct[model_num]['ATOM_ID'] = []
					lig_struct[model_num]['ATOM_NAME'] = []
				lig_struct[model_num]['COORD'].append([X_coord, Y_coord, Z_coord])
				lig_struct[model_num]['ATOM_ID'].append(atom_id)
				lig_struct[model_num]['ATOM_NAME'].append(atom_name)
				lig_struct[model_num]['ENERGY'] = model_energy
		else:
			continue
	fh.close()
	#print ("lig struct = ", lig_struct)
	return lig_struct		


def get_prot_lig_interactions(protein_dict, lig_dict, lig_name, output_dir, res_info, all_ligs_res_level_int_file, file_flag):
	int_dist = 4 # Pre-define the distance (in Ang) to define interaction between the heavy atoms of 
				 # the protein and ligand.
	lig_models = list(lig_dict.keys())
	lig_models.sort() # Sort the models numbers in increasing order
	

	# Get the coordinates, residue id and atom information of residues of interest
	all_res_coords = []
	all_res_uniq_ids = [] # create unique identifier for each residue by including the chain, res id and atom name
	all_res_uniq_names = []
	res_info_dict = dict() # We will store the KFC2 and Consurf scores for each residue of interest in this dictionary
	res_info_uniq_res_name_list = [] # To store the respective uniq residues names for the list of residues given in the res_info file
	for record in res_info:
		chain_id = record[0]
		res_id = record[1]
		res_kfc2_score = record[2]
		res_consurf_score = record[3]
		res_ind = list(np.where(np.array(protein_dict[chain_id]['RESID']) == int(res_id))[0])
		#print ("res ind = ", res_ind)
		if len(res_ind) == 0:
			print ("No matching residues found for ID ", res_id, " in ", res_info, " file\n");
			continue
		# Get the coordinates of the residue atom indices and also the unique identifier for each residue
		for ind_i in res_ind:
			coord_i = protein_dict[chain_id]['COORD'][ind_i]
			all_res_coords.append(coord_i)
			uniq_id = protein_dict[chain_id]['UNIQ_RES_ID'][ind_i]
			uniq_name = protein_dict[chain_id]['UNIQ_RES_NAME'][ind_i]
			all_res_uniq_ids.append(uniq_id)
			all_res_uniq_names.append(uniq_name)
			
			# Include information about the current residue's KFC2 score and Consurf score
			if uniq_name not in res_info_dict.keys():
				res_info_dict[uniq_name] = dict()
				res_info_dict[uniq_name]['KFC2_SCORE'] = float(res_kfc2_score)
				res_info_dict[uniq_name]['CONSURF_SCORE'] = float(res_consurf_score)
				res_info_uniq_res_name_list.append(uniq_name)
	
	# Open the all_ligs_res_level_int_file file in 'w' mode if file_flag is set to 0 otherwise, in 'a' mode
	if file_flag == 0:
		fh3 = open(all_ligs_res_level_int_file, 'w')
		res_contacts_str = ','.join(['NC_' + res_i for res_i in res_info_uniq_res_name_list])
		fh3.write('Ligand Name,Pose Num,' + res_contacts_str + ',Ligand Score,Contact Variance,Energy\n')
	else:
		fh3 = open(all_ligs_res_level_int_file, 'a')
	# Create two output files for a given ligand: 
	# 	1. Having the total number of interactions for a given pose
	#	2. Having details of interactions for each pose
	summary_file1 = output_dir + lig_name + '_int_summary.csv'
	details_file = output_dir + lig_name + '_int_details.txt'
	fh1 = open(summary_file1, 'w')
	fh2 = open(details_file, 'w')
	fh1.write('Ligand Pose, Number of contacts\n')
	
	num_interactions_list = [] # Store the number of interactions. We would want to calculate the median number of interactions later 
	num_poses = 0					       # for all the ligands.
	for model_i in lig_models:
		print ("Running calculations for ", lig_name, ", Pose ", model_i, "\n") 
		fh2.write('*** Interaction Details for Model ' + str(model_i) + '***\n')
		model_i_coords = lig_dict[model_i]['COORD']
		model_i_atom_ids = lig_dict[model_i]['ATOM_ID']
		model_i_atom_names = lig_dict[model_i]['ATOM_NAME']

		model_i_total_interactions = 0
		num_poses += 1
		pose_contacts_by_res = dict() # To store the contacts for a given model by respective residues
		for i in range(0,len(model_i_coords)):
			for j in range(0,len(all_res_coords)):
				coord_i = model_i_coords[i] # Get the coordinates for ith atom in the ligand
				coord_j = all_res_coords[j]
				lig_atom_id = model_i_atom_ids[i]
				lig_atom_name = model_i_atom_names[i]
				res_uniq_id = all_res_uniq_ids[j]
				lig_uniq_id = str(lig_atom_id) + '_' + lig_atom_name
				euc_norm = np.linalg.norm(np.array(coord_i)-np.array(coord_j)) # Calculate the euclidean norm
				if euc_norm <= int_dist: # Write the interaction details if conditions are satisfied
					model_i_total_interactions += 1
					fh2.write(lig_uniq_id + ' : ' + res_uniq_id + '\n')
					res_j_uniq_name = all_res_uniq_names[j]
					if res_j_uniq_name not in pose_contacts_by_res.keys():
						pose_contacts_by_res[res_j_uniq_name] = 1
					else:
						pose_contacts_by_res[res_j_uniq_name] += 1
		fh1.write(str(model_i) + ',' + str(model_i_total_interactions) + '\n')
		fh2.write("*** END ***\n\n");
		num_interactions_list.append(model_i_total_interactions)

		# Write the residue level-interactions into a separate file
		model_energy =	lig_dict[model_i]['ENERGY']
		contacts_by_res_list = []
		
		# Get the number of contacts per residue and weigh the contacts by the residue's KFC2 score and Consurf score
		weighted_contacts_list = []
		for res_name_i in res_info_uniq_res_name_list:
			# If the current model has contacts with res_name_i, then fetch the number of records from pose_contacts_by_res
			if res_name_i in pose_contacts_by_res.keys():
				res_i_num_contacts = pose_contacts_by_res[res_name_i]
			else: # Otherwise, assign the number of contacts with res_name_i as 0	
				res_i_num_contacts = 0
			kfc_score = res_info_dict[res_name_i]['KFC2_SCORE']
			consurf_score = res_info_dict[res_name_i]['CONSURF_SCORE']
			weighted_contacts_list.append(res_i_num_contacts*kfc_score*consurf_score) # Store the weighted contact information
		print ("lig name = ", lig_name, "model = ", model_i, "weighted contacts list = ", weighted_contacts_list,"\n")		
		contact_variance = np.var(weighted_contacts_list)
		lig_score = sum(weighted_contacts_list)
		weighted_contacts_str = ','.join([str(x) for x in weighted_contacts_list]) # join the contacts as a string after converting individual elements into strings
		fh3.write(lig_name + ',' + str(model_i) + ',' + weighted_contacts_str + ',' + str(lig_score) + ',' + str(contact_variance) + ',' + model_energy + '\n')
	fh1.close()
	fh2.close()
	fh3.close()

	median_num_interactions = statistics.median(num_interactions_list)
	total_interactions = sum(num_interactions_list)
	return median_num_interactions, num_poses, total_interactions


###### The main function #######################################
# The main function from which we will call other subroutines.
#
# Input args -
#	input_dir - Directory having the protein and ligand files. Must be a relative 
#				or full dir path. Must not end with '/'.
#
#	prot_file - Name of the PDB file of the protein (must have .pdb extension)
#	lig_file_prefix - A common prefix with which all ligand file names
#					start with (e.g. 'ligand_'). Must have extension of .pdb
#	res_file - A file have a set of residues and chains which are expected to
#			   interact with the ligand atoms. File should be in format:
#			   Chain ID, Res Name. 
#			   Can have multiple residues and multiple chain IDs.
#	output_dir - Directory where all the output files will be stored.
################################################################
def main(input_dir, prot_file, lig_file_prefix, res_file, output_dir):
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	if not output_dir.endswith('/'):
		output_dir = output_dir + '/'

	# The res_file must be in the following format
	# CHAIN_ID, RES_ID, KFC2_SCORE, CONSURF_SCORE
	res_info = np.loadtxt(input_dir + '/' + res_file, dtype='str', delimiter=',')
	prot_file_fq = input_dir + '/' + prot_file
	protein_dict = parse_protein_as_dict(prot_file_fq)
	current_dir = os.getcwd()
	os.chdir(input_dir)
	lig_files = glob.glob(lig_file_prefix + '*.pdb')
	median_interactions_file = output_dir + 'observed_2_expected_ratios.csv'
	all_ligs_res_level_int_file = output_dir + 'all_ligands_res_level_interactions.csv'
	if len(lig_files) == 0:
		sys.exit('No ligand files with prefix' + lig_file_prefix + ' found\n')
	else:
		print ("Found ", len(lig_files), " ligand files!\n")
		#print (lig_files)
		fh3 = open(median_interactions_file, 'w')
	num_poses_list = [] # Store the number of poses for each ligand
	num_interactions_list = [] # Store the number of interactions for each ligand
	file_flag = 0
	for lig_file_i in lig_files:
		print ('Running interaction calculations for ligand file ', lig_file_i)
		lig_file_i_fq = input_dir + '/' + lig_file_i
		lig_i_dict = parse_ligand_as_dict(lig_file_i_fq)
		[median_interactions, num_poses, num_interactions] = get_prot_lig_interactions(protein_dict, lig_i_dict, lig_file_i, output_dir, res_info, all_ligs_res_level_int_file, file_flag) # returns the median number of interactions across all poses of the given ligand
		file_flag = 1
		num_poses_list.append(num_poses)
		num_interactions_list.append(num_interactions)
		print ("Done!\n")
	print (sum(num_interactions_list))
	print (sum(num_poses_list))
	avg_int_per_pose = sum(num_interactions_list)/sum(num_poses_list)

	# Calculate the observed to expected ratio of contacts for each ligand file
	ind_i = 0;
	for lig_file_i in lig_files:
		obs_num_int = num_interactions_list[ind_i]
		num_poses = num_poses_list[ind_i]
		exp_num_int = avg_int_per_pose * num_poses
		obs_2_exp = obs_num_int/exp_num_int
		fh3.write(lig_file_i + ',' + str(obs_2_exp) + '\n')
		ind_i += 1
	fh3.close()



if __name__ == '__main__':
	usage = '\nUsage :\npython3.6 ' + sys.argv[0] + ' input_dir protein_file ligand_file_prefix res_file output_dir'
	if len(sys.argv) != 6:
		sys.exit('ERROR! Invalid number of arguments\n' + usage)
		exit
	else:
		input_dir = sys.argv[1]
		prot_file = sys.argv[2]
		lig_file_prefix = sys.argv[3]
		res_file = sys.argv[4]
		output_dir = sys.argv[5]

		#print("inputdir = ",input_dir, ", prot_file = ", prot_file, ", lig_file_prefix = ", lig_file_prefix, "res_file = ", res_file, "outputdir = ", output_dir)
		main(input_dir, prot_file, lig_file_prefix, res_file, output_dir)

