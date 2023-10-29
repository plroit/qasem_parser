<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/plroit/qasem_parser">
    <!-- This image was used originally in the paper: https://aclanthology.org/2021.emnlp-main.778.pdf -->
    <img src="images/logo.svg" alt="Logo">
  </a>

<h3 align="center">Parser for Question-Answer based Semantics</h3>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

Reimplementation of the [QA-SEM pipeline](https://github.com/kleinay/QASem/) with re-trained joint argument parser model

<!-- GETTING STARTED -->
## Getting Started

### Installation
   ```sh
   pip install qasem_parser
   ```

<!-- USAGE EXAMPLES -->
### Usage
```python
from typing import List
from qasem_parser import QasemParser, QasemFrame
arg_parser_path = "<argument parser model path>"
parser = QasemParser.from_pretrained(arg_parser_path)

sentences = [
    "The fox jumped over the fence.",
    "Back in May, a signal consistent with that of a radio beacon was detected in the area, but nothing turned up that helped with the search."
]
# frames is a list of lists, with one sublist per sentence such that len(frames) == len(sentences)  
# frames[i] is a sublist of the semantic frames that occur within sentence[i] 
frames: List[List[QasemFrame]] = parser(sentences)

# NOTE: if your text has already been tokenized and you want 
# the parser to respect the token boundaries, use the flag ```is_pretokenized=True``` 
sentences = [
    "Unfortunately , extensive property damage is bound to occur even with the best preparation .".split(),
    "Afghanistan to attend the summit after following the election in June , "
    "but the ongoing audit of votes has made this impossible .".split()
]
frames = parser(sentences, is_pretokenized=True)

for frames_per_sent in frames:
    # NOTE: frames_per_sent might be empty if no predicate is detected in the sentence.
    in_sent_predicates = " | ".join(f.predicate.text for f in frames_per_sent) 
    print(f"No of frames: {len(frames_per_sent)}, predicates: {in_sent_predicates}")
    frame = frames_per_sent[0]
    frame_args = [arg.text for arg in frame.arguments]
    print(frame.predicate.text, frame_args)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP 
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature
See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).
-->


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Paul Roit - [@paul_roit](https://twitter.com/paul_roit)

Project Link: [https://github.com/plroit/qasem_parser](https://github.com/plroit/qasem_parser)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Ayal Klein](https://github.com/kleinay)
* [Arie Cattan](https://ariecattan.github.io/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt

