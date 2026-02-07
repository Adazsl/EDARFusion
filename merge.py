import os

def merge_parts(parts, output_path):
    with open(output_path, 'wb') as outfile:
        for part in parts:
            part_path = os.path.join('checkpoint', part)
            with open(part_path, 'rb') as infile:
                outfile.write(infile.read())

merge_parts(['BSAFusion_CTaa', 'BSAFusion_CTab'], 'checkpoint/BSAFusion_CT.pkl')
merge_parts(['BSAFusion_PETaa', 'BSAFusion_PETab'], 'checkpoint/BSAFusion_PET.pkl')
merge_parts(['BSAFusion_SPECTaa', 'BSAFusion_SPECTab'], 'checkpoint/BSAFusion_SPECT.pkl')

