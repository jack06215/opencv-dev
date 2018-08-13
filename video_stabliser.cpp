#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::videostab;

#define arg(name) cmd.get<string>(name)
#define argb(name) cmd.get<bool>(name)
#define argi(name) cmd.get<int>(name)
#define argf(name) cmd.get<float>(name)
#define argd(name) cmd.get<double>(name)

string inputPath;
string outputPath;
const char *keys =
					// de-blurring
					"{  deblur                     | no | }"
					"{  r  radius                  | 15 | }"
					"{  deblur-sens                | 0.1 | }"
					// wobbling suppress
					"{  ws  wobble-suppress        | no | }"
					"{  ws-period                  | 30 | }"
					"{  ws-model                   | homography | }"
					"{  ws-subset                  | auto | }"
					"{  ws-thresh                  | auto | }"
					"{  ws-outlier-ratio           | 0.5 | }"
					"{  ws-min-inlier-ratio        | 0.1 | }"
					"{  ws-nkps                    | 1000 | }"
					"{  ws-local-outlier-rejection | no | }"
					// stablised video output directory
					"{ o  output                   | stabilized.avi | }";


MotionModel motionModel(const string &str)
{
	if (str == "trans")
		return MM_TRANSLATION;
	if (str == "trans&scale")
		return MM_TRANSLATION_AND_SCALE;
	if (str == "rigid")
		return MM_RIGID;
	if (str == "similarity")
		return MM_SIMILARITY;
	if (str == "affine")
		return MM_AFFINE;
	if (str == "homography")
		return MM_HOMOGRAPHY;
	throw runtime_error("unknown motion model: " + str);
}

class IMotionEstimatorBuilder
{
public:
	virtual ~IMotionEstimatorBuilder() {}
	virtual Ptr<ImageMotionEstimatorBase> build() = 0;
protected:
	IMotionEstimatorBuilder(CommandLineParser &command) : cmd(command) {}
	CommandLineParser cmd;
};


class MotionEstimatorRansacL2Builder : public IMotionEstimatorBuilder
{
public:
	MotionEstimatorRansacL2Builder(CommandLineParser &command, const string &_prefix = "")
		: IMotionEstimatorBuilder(command), prefix(_prefix) {}

	virtual Ptr<ImageMotionEstimatorBase> build() CV_OVERRIDE
	{
		Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(motionModel(arg(prefix + "model")));

		RansacParams ransac = est->ransacParams();
		if (arg(prefix + "subset") != "auto") ransac.size = argi(prefix + "subset");
		if (arg(prefix + "thresh") != "auto") ransac.thresh = argf(prefix + "thresh");
		ransac.eps = argf(prefix + "outlier-ratio");
		est->setRansacParams(ransac);
		est->setMinInlierRatio(argf(prefix + "min-inlier-ratio"));

		Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
		if (arg(prefix + "local-outlier-rejection") == "yes")
		{
			Ptr<TranslationBasedLocalOutlierRejector> tblor = makePtr<TranslationBasedLocalOutlierRejector>();
			RansacParams ransacParams = tblor->ransacParams();
			if (arg(prefix + "thresh") != "auto") ransacParams.thresh = argf(prefix + "thresh");
			tblor->setRansacParams(ransacParams);
			outlierRejector = tblor;
		}

		Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
		kbest->setDetector(GFTTDetector::create(argi(prefix + "nkps")));
		kbest->setOutlierRejector(outlierRejector);
		return kbest;
	}
private:
	string prefix;
};


// save video
void videoOutput(Ptr<IFrameSource> stabFrames, string outputPath)
{
	VideoWriter writer;
	cv::Mat stabFrame;
	int nframes = 0;
	
	// output video fps
	double outputFps = 30;
	while (!(stabFrame = stabFrames->nextFrame()).empty())
	{
		nframes++;
		if (!outputPath.empty())
		{
			if (!writer.isOpened())
				writer.open(outputPath, VideoWriter::fourcc('X', 'V', 'I', 'D'),
					outputFps, stabFrame.size());
			writer << stabFrame;
		}
		imshow("stabFrame", stabFrame);
		char key = static_cast<char>(waitKey(1));
		if (key == 27)
		{
			cout << endl;
			break;
		}
	}
	std::cout << "nFrames: " << nframes << endl;
	std::cout << "finished " << endl;
}

void run_videoStablise(Ptr<IFrameSource> stabFrames, string srcVideoFile, int argc, char* argv[])
{
	try
	{
		CommandLineParser cmd(argc, argv, keys);
		
		Ptr<VideoFileSource> srcVideo = makePtr<VideoFileSource>(inputPath);
		cout << "frame count: " << srcVideo->count() << endl;
		
		//	1. Setup MotionEstimatorRansacL2(cv::MotionModel MM_AFFINE, cv::RansacParams, double inlierRatio)
		Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(MM_AFFINE);
		RansacParams ransac = est->ransacParams();
		ransac.size = 3;
		ransac.thresh = 5;
		ransac.eps = 0.5;
		est->setRansacParams(ransac);
		est->setMinInlierRatio(0.1);
		
		
		// Setup KeypointBasedMotionEstimator(cv::cv::videostab::MotionEstimatorBase)
		Ptr<FastFeatureDetector> feature_detector = FastFeatureDetector::create();
		Ptr<KeypointBasedMotionEstimator> motionEstBuilder = makePtr<KeypointBasedMotionEstimator>(est);
		motionEstBuilder->setDetector(feature_detector);
		Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
		motionEstBuilder->setOutlierRejector(outlierRejector);
		
		// prepare the one or two pass stabilizer
		StabilizerBase *stabilizer = 0;
		bool isTwoPass = 1;
		int radius_pass = 15;
		Ptr<IMotionEstimatorBuilder> wsMotionEstBuilder;
		wsMotionEstBuilder.reset(new MotionEstimatorRansacL2Builder(cmd, "ws-"));
		if (isTwoPass)
		{
			// with a two pass stabilizer
			bool est_trim = true;
			TwoPassStabilizer *twoPassStabilizer = new TwoPassStabilizer();
			
			// wobble-suppress
			if (arg("wobble-suppress") == "yes")
			{
				Ptr<MoreAccurateMotionWobbleSuppressorBase> ws = makePtr<MoreAccurateMotionWobbleSuppressor>();
				ws->setMotionEstimator(wsMotionEstBuilder->build());
				ws->setPeriod(argi("ws-period"));
				twoPassStabilizer->setWobbleSuppressor(ws);
			}


			twoPassStabilizer->setEstimateTrimRatio(est_trim);
			twoPassStabilizer->setMotionStabilizer(makePtr<GaussianMotionFilter>(radius_pass));
			stabilizer = twoPassStabilizer;
		}
		else
		{
			// with an one pass stabilizer
			OnePassStabilizer *onePassStabilizer = new OnePassStabilizer();
			onePassStabilizer->setMotionFilter(makePtr<GaussianMotionFilter>(radius_pass));
			stabilizer = onePassStabilizer;
		}
		
		// set up the parameters
		stabilizer->setFrameSource(srcVideo);
		stabilizer->setMotionEstimator(motionEstBuilder);
		stabilizer->setRadius(15);
		stabilizer->setTrimRatio(0.1);
		stabilizer->setCorrectionForInclusion(false);
		stabilizer->setBorderMode(BORDER_REPLICATE);
		
		// cast stabilizer to simple frame source interface to read stabilized frames
		stabFrames.reset(dynamic_cast<IFrameSource*>(stabilizer));

		// init deblurer
		if (arg("deblur") == "yes")
		{
			Ptr<WeightingDeblurer> deblurer = makePtr<WeightingDeblurer>();
			deblurer->setRadius(argi("radius"));
			deblurer->setSensitivity(argf("deblur-sens"));
			stabilizer->setDeblurer(deblurer);
			std::cout << "deblur setup completed\n";
		}
		if (arg("output") != "no")
			outputPath = arg("output");

		// make the result (stablised) video.
		videoOutput(stabFrames, outputPath);
	}
	catch (const exception &e)
	{
		cout << "error: " << e.what() << endl;
		stabFrames.release();
	}
}
int main(int argc, char* argv[])
{
	Ptr<IFrameSource> stabFrames;
	inputPath = argv[0];
	if (inputPath.empty()) throw runtime_error("specify video path");
	run_videoStablise(stabFrames, inputPath, argc, argv);
	stabFrames.release();
	return 0;
}