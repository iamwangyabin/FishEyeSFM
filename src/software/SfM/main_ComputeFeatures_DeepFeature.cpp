// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// The <cereal/archives> headers are special and must be included first.
#include <cereal/archives/json.hpp>

#include "openMVG/image/image_io.hpp"
#include "openMVG/features/regions_factory_io.hpp"
#include "openMVG/sfm/sfm.hpp"
#include "openMVG/system/timer.hpp"


#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/progress/progress.hpp"

/// OpenCV Includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <cstdlib>
#include <fstream>
#include <vector>
#include <iostream>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>


using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::features;
using namespace openMVG::sfm;
using namespace std;

enum eGeometricModel
{
    FUNDAMENTAL_MATRIX = 0,
    ESSENTIAL_MATRIX   = 1,
    HOMOGRAPHY_MATRIX  = 2
};

enum ePairMode
{
    PAIR_EXHAUSTIVE = 0,
    PAIR_CONTIGUOUS = 1,
    PAIR_FROM_FILE  = 2
};


//- Create an Image_describer interface that use and OpenCV extraction method
// i.e. with the SIFT detector+descriptor
// Regions is the same as classic SIFT : 128 unsigned char
class LFNET_Image_describer : public Image_describer
{
public:
    using Regions_type = LFNET_Float_Regions;

    LFNET_Image_describer() : Image_describer() {}
    ~LFNET_Image_describer() {}

    bool Set_configuration_preset(EDESCRIBER_PRESET preset){
        return true;
    }

    std::unique_ptr<Regions> Describe(
            const image::Image<unsigned char>& image,
            const image::Image<unsigned char> * mask = nullptr
    ) override
    {
        return Describe_LFNET(image, mask);
    }

    std::unique_ptr<Regions_type> Describe_LFNET(
            const image::Image<unsigned char>& image,
            const image::Image<unsigned char>* mask = nullptr
    )
    {
        std::string filename_=filename;
        auto regions = std::unique_ptr<Regions_type>(new Regions_type);
        std::ifstream infile_feat(filename_);
        std::string feature;

        float feat_onePoint;  //存储每行按空格分开的每一个float数据
        std::vector<float> lines; //存储每行数据
        std::vector<vector<float>> lines_feat; //存储所有数据
        lines_feat.clear();
        while(!infile_feat.eof())
        {
            getline(infile_feat, feature); //一次读取一行数据

            stringstream stringin(feature); //使用串流实现对string的输入输出操作
            lines.clear();
            while (stringin >> feat_onePoint) {      //按空格一次读取一个数据存入feat_onePoint
                lines.push_back(feat_onePoint); //存储每行按空格分开的数据
            }
            if(lines.size() != 0){
                lines_feat.push_back(lines); //存储所有数据
            }
        }
        infile_feat.close();

        regions->Features().reserve(lines_feat.size());
        regions->Descriptors().reserve(lines_feat.size());

        // Copy keypoints and descriptors in the regions
        int cpt = 0;
        for (auto i_kp = lines_feat.begin();
             i_kp != lines_feat.end();
             ++i_kp, ++cpt)
        {
            SIOPointFeature feat((*i_kp)[0], (*i_kp)[1], 0, 0);
            regions->Features().push_back(feat);

            Descriptor<float, 256> desc;
            //
            for (int j = 0; j < 256; j++)
            {
                desc[j] = (*i_kp)[j+2];
            }
            //
            regions->Descriptors().push_back(desc);
        }

        return regions;
    };

    /// Allocate Regions type depending of the Image_describer
    std::unique_ptr<Regions> Allocate() const override
    {
        return std::unique_ptr<Regions_type>(new Regions_type);
    }

    template<class Archive>
    void serialize( Archive & ar )
    {
    }
};
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>

CEREAL_REGISTER_TYPE_WITH_NAME(LFNET_Image_describer, "LFNET_Image_describer");
CEREAL_REGISTER_POLYMORPHIC_RELATION(openMVG::features::Image_describer, LFNET_Image_describer)





int main(int argc, char **argv) {
    CmdLine cmd;

    std::string sSfM_Data_Filename;
    std::string sOutDir = "";
    bool bForce = false;
    std::string sImage_Describer_Method = "LFNET";
    std::string feats_dir = "empty";

    cmd.add(make_option('i', sSfM_Data_Filename, "input_file"));
    cmd.add(make_option('o', sOutDir, "outdir"));
    cmd.add(make_option('f', bForce, "force"));
    cmd.add(make_option('m', sImage_Describer_Method, "describerMethod"));
    cmd.add(make_option('d', feats_dir, "feats_dir"));


    try {
        if (argc == 1) throw std::string("Invalid command line parameter.");
        cmd.process(argc, argv);
    } catch (const std::string &s) {
        std::cerr << "Usage: " << argv[0] << '\n'
                  << "[-i|--input_file]: a SfM_Data file \n"
                  << "[-o|--outdir] path \n"
                  << "\n[Optional]\n"
                  << "[-f|--force] Force to recompute data\n"
                  << "[-m|--describerMethod]\n"
                  << "  (method to use to describe an image):\n"
                  << "   LFNET (default),\n"
                  << "[-d|--featsdir\n"
                  << std::endl;

        std::cerr << s << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << " You called : " << std::endl
              << argv[0] << std::endl
              << "--input_file " << sSfM_Data_Filename << std::endl
              << "--outdir " << sOutDir << std::endl
              << "--describerMethod " << sImage_Describer_Method << std::endl
              << "--force " << bForce << std::endl
              << "--feats_dir " << feats_dir << std::endl;

    if (sOutDir.empty()) {
        std::cerr << "\nIt is an invalid output directory" << std::endl;
        return EXIT_FAILURE;
    }

    // Create output dir
    if (!stlplus::folder_exists(sOutDir)) {
        if (!stlplus::folder_create(sOutDir)) {
            std::cerr << "Cannot create output directory" << std::endl;
            return EXIT_FAILURE;
        }
    }

    //---------------------------------------
    // a. Load input scene
    //---------------------------------------
    SfM_Data sfm_data;
    if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS | INTRINSICS))) {
        std::cerr << std::endl
                  << "The input file \"" << sSfM_Data_Filename << "\" cannot be read" << std::endl;
        return false;
    }

    // Init the image_describer
    // - retrieve the used one in case of pre-computed features
    // - else create the desired one

    using namespace openMVG::features;
    std::unique_ptr<Image_describer> image_describer;

    const std::string sImage_describer = stlplus::create_filespec(sOutDir, "image_describer", "json");

    image_describer.reset(new LFNET_Image_describer);
    {
        std::ofstream stream(sImage_describer.c_str());
        if (!stream.is_open())
            return false;
        cereal::JSONOutputArchive archive(stream);
        archive(cereal::make_nvp("image_describer", image_describer));
        auto regions = image_describer->Allocate();
        archive(cereal::make_nvp("regions_type", regions));
    }

    {
        system::Timer timer;
        Image<unsigned char> imageGray;

        C_Progress_display my_progress_bar(sfm_data.GetViews().size(),
                                           std::cout, "\n- EXTRACT FEATURES -\n");

        // Use a boolean to track if we must stop feature extraction
        bool preemptive_exit(false);
        for (auto iterViews = sfm_data.views.cbegin();
             iterViews != sfm_data.views.cend() && !preemptive_exit;
             ++iterViews) {
            const View *view = iterViews->second.get();
            const std::string
                    sView_filename = stlplus::create_filespec(sfm_data.s_root_path, view->s_Img_path),
                    sFeat = stlplus::create_filespec(sOutDir, stlplus::basename_part(sView_filename), "feat"),
                    sDesc = stlplus::create_filespec(sOutDir, stlplus::basename_part(sView_filename), "desc");

            if (bForce || !stlplus::file_exists(sFeat) || !stlplus::file_exists(sDesc)) {
                if (!ReadImage(sView_filename.c_str(), &imageGray))
                    continue;

                Image<unsigned char> *mask = nullptr; // The mask is null by default

                const std::string
                        mask_filename_local =
                        stlplus::create_filespec(sfm_data.s_root_path,
                                                 stlplus::basename_part(sView_filename) + "_mask", "png"),
                        mask__filename_global =
                        stlplus::create_filespec(sfm_data.s_root_path, "mask", "png");

                Image<unsigned char> imageMask;
                // Try to read the local mask
                if (stlplus::file_exists(mask_filename_local)) {
                    if (!ReadImage(mask_filename_local.c_str(), &imageMask)) {
                        std::cerr << "Invalid mask: " << mask_filename_local << std::endl
                                  << "Stopping feature extraction." << std::endl;
                        preemptive_exit = true;
                        continue;
                    }
                    // Use the local mask only if it fits the current image size
                    if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
                        mask = &imageMask;
                } else {
                    // Try to read the global mask
                    if (stlplus::file_exists(mask__filename_global)) {
                        if (!ReadImage(mask__filename_global.c_str(), &imageMask)) {
                            std::cerr << "Invalid mask: " << mask__filename_global << std::endl
                                      << "Stopping feature extraction." << std::endl;
                            preemptive_exit = true;
                            continue;
                        }
                        // Use the global mask only if it fits the current image size
                        if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
                            mask = &imageMask;
                    }

                    vector<string> vStr;
                    boost::split(vStr, sView_filename, boost::is_any_of("/"), boost::token_compress_on);
                    const std::string featpath = stlplus::create_filespec(feats_dir, *(vStr.end() - 1), ".txt");
                    image_describer->getimgName(featpath);

                    auto regions = image_describer->Describe(imageGray, mask);

                    if (regions && !image_describer->Save(regions.get(), sFeat, sDesc)) {
                        std::cerr << "Cannot save regions for images: " << sView_filename << std::endl
                                  << "Stopping feature extraction." << std::endl;
                        preemptive_exit = true;
                        continue;
                    }
                    ++my_progress_bar;
                }
            }
            std::cout << "Task done in (s): " << timer.elapsed() << std::endl;
        }
        return EXIT_SUCCESS;
    }
}
