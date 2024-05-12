import spacy
import inspect
import textwrap
import nodes
import re
import random
import itertools
from functools import reduce
import unicodedata
import emoji

class AnyType(str):

    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

nlp = spacy.load("en_core_web_sm")


class SrlConditionalInterrupt:
    """Interrupt processing if the boolean input is true. Pass through the other input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interrupt": ("BOOLEAN", {
                    "forceInput": True
                }),
                "inp": (any_typ, ),
            },
        }

    RETURN_TYPES = (any_typ, )
    RETURN_NAMES = ("output", )
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, interrupt, inp):
        if interrupt:
            nodes.interrupt_processing()

        return (inp, )


class SrlFormatString:
    """Use Python f-string syntax to generate a string using the inputs as the arguments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": (
                    "STRING",
                    {
                        "multiline":
                        False,
                        "default":
                        "first input via str(): {}, second input via repr(): {!r}, third input by index: {2}, fifth input by name: {in4}",
                    },
                ),
            },
            "optional": {
                "in0": (any_typ, ),
                "in1": (any_typ, ),
                "in2": (any_typ, ),
                "in3": (any_typ, ),
                "in4": (any_typ, ),
            },
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, format, **kwargs):
        # Allow referencing arguments both by name and index.
        return (format.format(*kwargs.values(), **kwargs), )


class SrlEval:
    """Evaluate any Python code as a function with the given inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parameters": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": 'a, b=None, c="foo", *rest',
                        "dynamicPrompts": False,
                    },
                ),
                "code": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "code goes here\nreturn a + b",
                        "dynamicPrompts": False,
                    },
                ),
            },
            "optional": {
                "arg0": (any_typ, ),
                "arg1": (any_typ, ),
                "arg2": (any_typ, ),
                "arg3": (any_typ, ),
                "arg4": (any_typ, ),
            },
        }

    RETURN_TYPES = (any_typ, )
    FUNCTION = "doit"
    CATEGORY = "utils"



    def doit(self, parameters, code, **kw):
        # Indent the code for the main body of the function
        func_code = textwrap.indent(code, "    ")
        source = f"def func({parameters}):\n{func_code}"

        # The provided code can mutate globals or really do anything, but ComfyUI isn't secure to begin with.
        loc = {}
        exec(source, globals(), loc)
        func = loc["func"]

        argspec = inspect.getfullargspec(func)
        # We don't allow variable keyword arguments or keyword only arguments, but we do allow varargs
        assert argspec.varkw is None
        assert not argspec.kwonlyargs

        input_names = list(self.INPUT_TYPES()["optional"].keys())
        parameter_names = argspec.args

        # Convert the list of defaults into a dictionary to make it easier to use
        default_list = argspec.defaults if argspec.defaults is not None else []
        defaults = {
            parameter_name: default
            for parameter_name, default in zip(
                parameter_names[-len(default_list):], default_list)
        }

        # We handle substituting default values ourselves in order to support *args
        args = [
            kw[input_name] if input_name in kw else defaults[parameter_name]
            for parameter_name, input_name in zip(parameter_names, input_names)
        ]

        # Support *args
        if argspec.varargs is not None:
            unnamed_inputs = input_names[len(argspec.args):]
            # I considered requiring the remaining inputs to be contiguous, but I don't think it's helpful.
            args += [
                kw[input_name] for input_name in unnamed_inputs
                if input_name in kw
            ]

        ret = func(*args)
        return (ret, )


class SrlFilterImageList:
    """Filter an image list based on a list of bools"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "keep": ("BOOLEAN", {
                    "forceInput": True
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "doit"

    def doit(self, images, keep):
        return ([im for im, k in zip(images, keep) if k], )


class SrlRandomizeAndFormatString:
    """
    This class replaces specified characters and patterns in a string with commas,
    then randomizes the order of the resulting tokens, converts them to lowercase,
    and finally joins them back into a comma-separated string without extra spaces.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {
                    "multiline": True
                }),
                "and_string": ("STRING", {
                    "default": ",",
                    "multiline": False
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF
                }),
                "max_length": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 999999999
                }),
            },
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "doit"
    CATEGORY = "utils"


    def recursive_replace(self, processed_string, words_to_replace):
        while True:
            old_processed_string = processed_string
            for word in words_to_replace:
                # Only replace if the word is surrounded by spaces or has a space before it
                processed_string = re.sub(re.escape(word), ",", processed_string)
            if old_processed_string == processed_string:
                break

        # Remove consecutive commas
        processed_string = re.sub(",{2,}", ",", processed_string)

        # Split the string into a list of words
        words = processed_string.split(',')

        # Randomize the order of the words
        random.shuffle(words)

        # Join the words back together into a string
        processed_string = ','.join(words)

        return processed_string

    def remove_stop_words(self, text, and_string=","):
        """Removes stop words from the input text using spaCy."""
        AND_STRING = and_string
        filtered_text = text.replace(" - ", AND_STRING)
        filtered_text = filtered_text.replace(" -", AND_STRING)
        filtered_text = filtered_text.replace("- ", AND_STRING)
        filtered_text = filtered_text.replace(" -", AND_STRING)
        return filtered_text

    def process_string(self, input_string):
        replacements = {
            "_": " ",
            r"\d+\.(?!\d)": ",",
            r"(?<!\d)\.|\.(?!\d)": ",",
            r"\:(?!\d)": "",
            r"(?<!\w)'(?!\w)": ",",
            r"(?<!\w)-(?!\w)": ",",
            r"\((?![^()]*:)[^()]*\)": ",",
            "```": ",",
            "jsonl": ",",
            ";": ",",
            "!": ",",
            "?": ",",
            ": ": " ",
            "“": ",",
            "”": ",",
            "‘": ",",
            "’": ",",
            "s'": "s",
            "' ": ",",
            " '": ",",
            "/": ",",
            "\\": ",",
            "[": ",",
            "]": ",",
            "{": ",",
            "}": ",",
            "<": "",
            ">": "",
            "=": "",
            "^": "",
            "|": ",",
            ", ": ",",
            "--": ",",
            " - ": ",",
            "---": ",",
            "- ": ",",
            " -": ",",
            "…": ",",
            "=>": ",",
            "->": ",",
            "\n": ",",
            '"': ",",
            "#": ",",
            "@": ",",
            "$": ",",
            "%": ",",
            "&": ",",
            "*": ",",
            "+": ",",
            "...": ",",
            ".": ","
        }

        for pattern, replacement in replacements.items():
            if pattern.startswith("\\") and len(
                    pattern) > 6:  # if it's a regex pattern
                input_string = re.sub(pattern,
                                    str(replacement),
                                    input_string,
                                    flags=re.UNICODE)
            else:  # if it's a simple string
                input_string = input_string.replace(pattern, replacement)

        return input_string

    def remove_emoji(self, text: str) -> str:
        """
        Removes emoji characters from the input text.

        Args:
        text (str): The input text containing emoji characters.

        Returns:
        str: The input text with emoji characters removed.
        """
        return ''.join(char for char in text if not emoji.is_emoji(char))

    def doit(self, input_string, and_string, max_length=1024, seed=0):
        """Cleans, removes stop words, tokenizes, shuffles, and selectively recombines tokens."""
        random.seed(seed)
        # "," and then remove all commas
        input_string = input_string.replace(",", "")

        # Preserve periods that are part of a number and replace others with ","
        preserved_periods = re.sub(r"\d+\.(?!\d)", ",", input_string)
        preserved_periods = re.sub(r"(?<!\d)\.|\.(?!\d)", ",", preserved_periods)

        # Replace colons that stand alone with ','
        preserved_periods = re.sub(r"(?<=\s):(?=\s)", ",", preserved_periods)

        # Replace colons with an alpha character before but not a number character after with ' '
        preserved_periods = re.sub(r"(?<=[a-zA-Z]):(?!\d)", " ", preserved_periods)

        # Replace remaining colons with comma
        preserved_periods = preserved_periods.replace(":", ",")
        preserved_periods = preserved_periods.replace("s'", "s")
        # Replace standalone apostrophes, hyphens, and text within parentheses with ","
        preserved_periods = re.sub(r"(?<!\w)'(?!\w)", ",", preserved_periods)
        preserved_periods = re.sub(r"(?<!\w)-(?!\w)", ",", preserved_periods)
        preserved_periods = re.sub(r"(?<!\s)'(?!\s)", "", preserved_periods)
        preserved_periods = re.sub(r"(?<!\s)-(?!\s)", "", preserved_periods)

        # Replace "(" and ")" with "," if there is no colon followed by a number within
        preserved_periods = re.sub(r"\(([^():]*)(?![^()]*:\d)[^()]*\)", ",", preserved_periods)

        # Create processed_string from preserved_periods for further replacements
        processed_string = preserved_periods

        # Replace various punctuation and symbols
        processed_string = processed_string.replace(";", ",")
        processed_string = processed_string.replace("!", ",")
        processed_string = processed_string.replace("?", ",")
        processed_string = processed_string.replace("“", ",")
        processed_string = processed_string.replace("”", ",")
        processed_string = processed_string.replace('"', ",")
        processed_string = processed_string.replace("‘", ",")
        processed_string = processed_string.replace("’", ",")
        processed_string = processed_string.replace("[", ",")
        processed_string = processed_string.replace("]", ",")
        processed_string = processed_string.replace("{", ",")
        processed_string = processed_string.replace("}", ",")
        processed_string = processed_string.replace("|", ",")
        processed_string = processed_string.replace("#", ",")
        processed_string = processed_string.replace("@", ",")
        processed_string = processed_string.replace("$", ",")
        processed_string = processed_string.replace("%", ",")
        processed_string = processed_string.replace("&", ",")
        processed_string = processed_string.replace("*", ",")
        processed_string = processed_string.replace(" - ", ",")
        processed_string = processed_string.replace("---", ",")
        processed_string = processed_string.replace("--", ",")
        processed_string = processed_string.replace("- ", ",")
        processed_string = processed_string.replace(" -", ",")
        processed_string = processed_string.replace("…", ",")
        processed_string = processed_string.replace("...", ",")
        processed_string = processed_string.replace("..", ",")
        processed_string = processed_string.replace(".", ",")
        processed_string = processed_string.replace("=>", ",")
        processed_string = processed_string.replace("->", ",")
        processed_string = processed_string.replace('\n', ",")
        processed_string = processed_string.replace("' ", ",")
        processed_string = processed_string.replace(" '", ",")
        processed_string = processed_string.replace("/", "")
        processed_string = processed_string.replace('\\', "")
        processed_string = processed_string.replace("<", "")
        processed_string = processed_string.replace(">", "")
        processed_string = processed_string.replace("=", "")
        processed_string = processed_string.replace("+", "")
        processed_string = processed_string.replace("_", " ")
        processed_string = processed_string.replace("^", "")
        
        # Remove emojis and stop words
        processed_string = self.remove_emoji(processed_string)
        processed_string = self.remove_stop_words(processed_string, and_string)

        # Final cleanup to remove extra spaces
        processed_string = re.sub(r" {2,}", " ", processed_string)
        processed_string = re.sub(r"\s+", " ", processed_string)
        words_to_replace = [
            "Here are the","JSONL entries that merge", "join the","provided keywords into","a", "unique visual description", ",ed", ", ed", " ed,",
            "here is the output", "here is the output of", "output",
            "micro level expanded visual representation",
            "micro level expanded visual", "expanded visual", "expanded",
            "visual", "expanded visual representation", "representation",
            "micro level representation", "micro level visual", "micro level",
            "micro", "level", "expanded representation", "keyword", "keywords",
            "main keyword", "main keywords", "tags", "tag", "interesting tag",
            "interesting tags", "sentence", "sentence fragment", "detail",
            "details", "micro level", "interesting tags", "sentence",
            "sentence fragment", "detail", "details", "micro level",
            "micro level details", "micro details", "components", "prompt",
            "input", "dall-e prompt", "dall-e", "\\", "dall-e input",
            "dall-e prompt", "dall-e input", "dall-e prompt", "comma",
            "comma separated", "comma separated list",
            "comma separated list of", "/}"
        ]

        # Sort the list in descending order by length
        words_to_replace.sort(key=len, reverse=True)

        processed_string = self.recursive_replace(processed_string,
                                                  words_to_replace)
        tokens = [
            token.strip().lower() for token in processed_string.split(",")
            if token.strip()
        ]

        # Remove stop words for each token and strip afterwards
        tokens = [self.remove_stop_words(token).strip() for token in tokens]

        print(f"Tokens before shuffling: {tokens}")
        shuffled_tokens = random.sample(tokens, len(tokens))
        print(f"Tokens after shuffling: {shuffled_tokens}")
        tokens_with_parentheses = [
            token for token in tokens if "(" in token and ")" in token
        ]
        tokens_without_parentheses = [
            token for token in tokens if token not in tokens_with_parentheses
        ]

        # Ensure there are no spaces after "(" or before ")"
        tokens_with_parentheses = [
            token.replace(" (", "(").replace(") ", ")")
            for token in tokens_with_parentheses
        ]
        
        # Filter out tokens longer than 65 characters or shorter than 3 characters
        tokens_with_parentheses = [token for token in tokens_with_parentheses if 3 <= len(token) <= 65]
        tokens_without_parentheses = [token for token in tokens_without_parentheses if 3 <= len(token) <= 65]

        # Shuffle tokens without parentheses
        random.shuffle(tokens_without_parentheses)

        # Calculate the total length of tokens with parentheses
        total_length_with_parentheses = sum(
            len(token) for token in tokens_with_parentheses)

        # Select tokens without parentheses until the total length of all tokens is at least 2048
        selected_tokens_without_parentheses = []
        total_length = total_length_with_parentheses
        for token in tokens_without_parentheses:
            if total_length >= max_length:
                break
            selected_tokens_without_parentheses.append(token)
            total_length += len(token)

        # Combine the two sets of tokens
        final_tokens = tokens_with_parentheses + selected_tokens_without_parentheses
        random.shuffle(final_tokens)

        # Filter out empty strings and join into a string
        output_string = ", ".join(token for token in final_tokens if token)
        
        # Remove leading and trailing spaces and commas
        output_string = output_string.strip(", ")
        print(f"Final output string: {output_string}")
        return (output_string, )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SRL Conditional Interrrupt": SrlConditionalInterrupt,
    "SRL Format String": SrlFormatString,
    "SRL Eval": SrlEval,
    "SRL Filter Image List": SrlFilterImageList,
}
NODE_CLASS_MAPPINGS[
    "SRL Randomize And Format String"] = SrlRandomizeAndFormatString

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SrlConditionalInterrupt": "SRL Conditional Interrupt",
    "SrlFormatString": "SRL Format String",
    "SrlEval": "SRL Eval",
    "SrlFilterImageList": "SRL Filter Image List",
}
NODE_DISPLAY_NAME_MAPPINGS[
    "SrlRandomizeAndFormatString"] = "SRL Randomize And Format String"
