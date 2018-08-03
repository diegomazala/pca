#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include "tinyply.h"
#include "pca.h"
#include <regex>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;



// list of paths of all files under the directory 'dir' when the extension matches the regex
// file_list<true> searches recursively into sub-directories; 
// file_list<false> searches only the specified directory
template <bool RECURSIVE> std::vector<fs::path> file_list(fs::path dir, std::regex ext_pattern)
{
	std::vector<fs::path> result;
	using iterator = std::conditional<RECURSIVE, fs::recursive_directory_iterator, fs::directory_iterator>::type;
	const iterator end;
	for (iterator iter{ dir }; iter != end; ++iter)
	{
		const std::string extension = iter->path().extension().string();
		if (fs::is_regular_file(*iter) && std::regex_match(extension, ext_pattern)) result.push_back(*iter);
	}
	return result;
}


int main(int argc, char* argv[])
{

	std::cout
		<< std::fixed << std::endl
		<< "Usage            : ./<app.exe> <dir>" << std::endl
		<< "Default          : ./pca_mesh.exe ../../data/" << std::endl
		<< std::endl;

	//
	// Initial parameters
	//
	const fs::path input_dir = (argc > 1) ? argv[1] : "../../data/";
	const std::string output_filename = "output_pca.ply";
	const std::vector<fs::path>& input_files = file_list<false>(input_dir, std::regex("\\.(?:ply)"));


	// 
	// Compose output filename
	//
	std::stringstream output_abs_filename;
	output_abs_filename << input_dir << "/Output/";
	fs::create_directory(output_abs_filename.str());
	output_abs_filename << output_filename;
	
	//
	// Verify the first file and check the number of vertices
	//
	std::ifstream ss(input_files.at(0).string(), std::ios::binary);
	tinyply::PlyFile file(ss);
	std::vector<float> verts_first_file;
	file.request_properties_from_element("vertex", { "x", "y", "z" }, verts_first_file);

	//
	// Create matrix for PCA
	// 
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> pca_input_matrix(verts_first_file.size(), input_files.size());

	std::cout << "Matrix size: " << pca_input_matrix.rows() << ' ' << pca_input_matrix.cols() << std::endl << std::endl;


	std::vector<float> verts;
	std::vector<float> norms;
	std::vector<uint8_t> colors;
	std::vector<uint32_t> faces;
	std::vector<float> uvCoords;
	uint32_t file_count = 0;

	for (const auto& filename : input_files)
	{
		try
		{
			std::cout << "Reading file <" << filename.string() << "> ... ";
			// 
			// Read source ply file
			//
			std::ifstream ss(filename.string(), std::ios::binary);
			tinyply::PlyFile file(ss);


			uint32_t vertexCount, normalCount, colorCount, faceCount, faceTexcoordCount, faceColorCount;
			vertexCount = normalCount = colorCount = faceCount = faceTexcoordCount = faceColorCount = 0;

			vertexCount = file.request_properties_from_element("vertex", { "x", "y", "z" }, verts);

			if (verts.size() != verts_first_file.size())
				throw("[FAIL] The number of vertices does not match");

			normalCount = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }, norms);
			colorCount = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }, colors);

			faceCount = file.request_properties_from_element("face", { "vertex_indices" }, faces, 3);
			faceTexcoordCount = file.request_properties_from_element("face", { "texcoord" }, uvCoords, 6);

			file.read(ss);

			for (int i = 0; i < verts.size(); ++i)
			{
				pca_input_matrix(i, file_count) = verts[i];
			}
			
			++file_count;

			std::cout << "verts: " << vertexCount << " [OK]" << std::endl;
		}
		catch (const std::exception & e)
		{
			std::cerr << "Caught exception: " << e.what() << std::endl;
		}
	}
	


	pca_t<float> pca;
	pca.set_input(pca_input_matrix);
	pca.compute();

	std::cout
		<< "Values: \n" << pca.get_eigen_values() << std::endl << std::endl
		<< "Vectors: \n" << pca.get_eigen_vectors() << std::endl << std::endl;

	const auto& result = pca.reprojection();

	for (int i = 0; i < verts.size(); ++i)
	{
		verts[i] = result(i, 0);
	}

	try
	{
		
		//
		// Write ply file
		//
		std::filebuf fb;
		fb.open(output_abs_filename.str(), std::ios::out | std::ios::binary);
		std::ostream outputStream(&fb);

		tinyply::PlyFile ply_out_file;

		if (!verts.empty())
			ply_out_file.add_properties_to_element("vertex", { "x", "y", "z" }, verts);
		if (!norms.empty())
			ply_out_file.add_properties_to_element("vertex", { "nx", "ny", "nz" }, norms);
		if (!colors.empty())
			ply_out_file.add_properties_to_element("vertex", { "red", "green", "blue", "alpha" }, colors);
		if (!faces.empty())
			ply_out_file.add_properties_to_element("face", { "vertex_indices" }, faces, 3, tinyply::PlyProperty::Type::UINT8);
		if (!uvCoords.empty())
			ply_out_file.add_properties_to_element("face", { "texcoord" }, uvCoords, 6, tinyply::PlyProperty::Type::UINT8);

		ply_out_file.write(outputStream, true);

		fb.close();
	}
	catch (const std::exception & e)
	{
		std::cerr << "Caught exception: " << e.what() << std::endl;
	}

}