# pyscanner
Scan documents from pictures in Python using [OpenCV](https://opencv.org/).

# Usage

```
$ python pyscanner.py "image_path" "destination_path"
``` 

If the document was not correctly scanned there is a way to find where the process failed. Just run:
```
$ python pyscanner.py --debug "image_path" "destination_path"
``` 
Along the final (although incorrect) scanned document there will be an additional file with the original image with the detected document's corners and edges -- red dots and lines -- and all detected edges in the image -- blue lines --. 
