# MovieLens 20M YouTube Trailers Dataset

This amendment to the [MovieLens 20M Dataset][0], YouTube Trailers, is a CSV file that maps MovieLens Movie IDs to YouTube IDs representing movie trailers. It was created to standardize a Computer Vision dataset as part of MovieLens-20M. We hope that this amendment will be used by machine learning researchers, in conjunction with the ratings.csv file of the MovieLens 20M dataset, for researching approaches that jointly use audio-visual and collaborative filtering information.


## Data Collection

We queried www.google.com to obtain YouTube video IDs for the trailers of MovieLens-20M movies. Specifically, we used [Google's XML API][1] and constructed a query of "<T> (<Y>) trailer", for each MovieLens movie, where <T> and <Y>, respectively, are the title and the release year of the movie. We processed Google's search response, taking the first URL starting with "www.youtube.com/watch?v=" within the first five search results. We found that, with manual inspection, Google searches consistently return the correct trailer with a higher accuracy than YouTube searches.

Out of the 27,278 unique movie IDs used in MovieLens-20M, our method was able to retrieve the YouTube IDs of 25,623 trailers (= 0.94 hit rate). The trailers have not all been manually verified.

This data was collected in May, 2017.


## Citation

If you use the dataset, we ask you to cite:

    @inproceedings{ml_yt_trailers,
        title={MovieLens 20M YouTube Trailers Dataset},
        author={Sami Abu-El-Haija and Joonseok Lee and Max Harper and Joseph Konstan},
        booktitle={MovieLens},
        year={2018},
    }


## Data File Structure (ml-20m-youtube.csv)

All trailer IDs are contained in the file ml-20m-youtube.csv. Each line of this file after the header row represents one YouTube ID and MovieLens movie ID pair, and has the following format:

    youtubeId,movieId,title

The lines within this file are ordered by movieId.  YouTube IDs map to URLs using the template <https://www.youtube.com/watch?v=ID>.


## License

This dataset is licenced under CC BY 4.0. This license lets others distribute, remix, tweak, and build upon this work, even commercially, as long as they credit you the authors for the original creation by using the citation above.  License link: <https://creativecommons.org/licenses/by/4.0/>


## References

[0]: <https://grouplens.org/datasets/movielens/20m/> "MovieLens 20M Dataset"

[1]: <https://developers.google.com/custom-search/docs/xml_results> "XML API reference, Google Custom Search"

