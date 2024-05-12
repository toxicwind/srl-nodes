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
import logging


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
                processed_string = re.sub(re.escape(word), ",",
                                          processed_string)
            if old_processed_string == processed_string:
                break
        processed_string = re.sub(",{2,}", ",", processed_string)
        words = processed_string.split(',')
        random.shuffle(words)
        processed_string = ','.join(words)
        return processed_string

    def remove_stop_words(self, text, and_string=","):
        AND_STRING = and_string
        text = text.replace(" - ", AND_STRING)
        text = text.replace(" -", AND_STRING)
        text = text.replace("- ", AND_STRING)
        text = text.replace(" -", AND_STRING)
        return text

    def remove_emoji(self, text: str) -> str:
        """
        Removes emoji characters from the input text.

        Args:
        text (str): The input text containing emoji characters.

        Returns:
        str: The input text with emoji characters removed.
        """
        return ''.join(char for char in text if not emoji.is_emoji(char))

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

        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        def recursive_replace(self, processed_string, words_to_replace):
            while True:
                old_processed_string = processed_string
                for word in words_to_replace:
                    processed_string = re.sub(re.escape(word), ",", processed_string)
                if old_processed_string == processed_string:
                    break
            processed_string = re.sub(",{2,}", ",", processed_string)
            words = processed_string.split(',')
            random.shuffle(words)
            processed_string = ','.join(words)
            return processed_string

        def remove_stop_words(self, text, and_string=","):
            AND_STRING = and_string
            text = text.replace(" - ", AND_STRING)
            text = text.replace(" -", AND_STRING)
            text = text.replace("- ", AND_STRING)
            text = text.replace(" -", AND_STRING)
            return text

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
            self.logger.debug("Initial input string: %s", input_string)
            random.seed(seed)
            input_string = input_string.lower()
            words_to_replace = [
                "Here is the", "Here is the output", "Here is the output of",
                "Here are the", "Here are", "JSONL entries", "that merge",
                "join the", "provided keywords", "Here are the",
                "JSONL entries that merge", "join the", "provided keywords into",
                "unique visual description", ",ed", ", ed", " ed,",
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
                "input", "dall-e prompt", "dall-e", "\\\\", "dall-e input",
                "dall-e prompt", "dall-e input", "dall-e prompt", "comma",
                "comma separated", "comma separated list",
                "comma separated list of"
            ]

            words_to_replace = [word.lower() for word in words_to_replace]
            words_to_replace.sort(key=len, reverse=True)
            self.logger.debug("Words to replace: %s", words_to_replace[)
            input_string = self.recursive_replace(input_string, words_to_replace)
            self.logger.debug("After recursive_replace: %s", input_string)
            input_string = input_string.replace(",", "")
            self.logger.debug("After lowercase and comma replacement: %s", input_string)
            preserved_periods = re.sub(r"\d+\.(?!\d)", ",", input_string)
            preserved_periods = re.sub(r"(?<!\d)\.|\.(?!\d)", ",", preserved_periods)
            preserved_periods = re.sub(r"(?<=\s):(?=\s)", ",", preserved_periods)
            preserved_periods = re.sub(r"(?<=[a-zA-Z]):(?!\d)", " ", preserved_periods)
            preserved_periods = preserved_periods.replace(":", ",")
            preserved_periods = preserved_periods.replace("s'", "s")
            preserved_periods = re.sub(r"(?<!\w)'(?!\w)", ",", preserved_periods)
            preserved_periods = re.sub(r"(?<!\w)-(?!\w)", ",", preserved_periods)
            preserved_periods = re.sub(r"(?<!\s)'(?!\s)", "", preserved_periods)
            preserved_periods = re.sub(r"(?<!\s)-(?!\s)", "", preserved_periods)
            preserved_periods = re.sub(r"\(([^():]*)(?![^()]*:\d)[^()]*\)", ",", preserved_periods)
            processed_string = preserved_periods
            self.logger.debug("After period replacements: %s", processed_string)
            processed_string = self.remove_stop_words(processed_string, and_string)
            processed_string = self.remove_emoji(processed_string)
            self.logger.debug("After remove_stop_words: %s", processed_string)
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
            processed_string = processed_string.replace('\\n', ",")
            processed_string = processed_string.replace("' ", ",")
            processed_string = processed_string.replace(" '", ",")
            processed_string = processed_string.replace("/", "")
            processed_string = processed_string.replace('\\\\', "")
            processed_string = processed_string.replace("<", "")
            processed_string = processed_string.replace(">", "")
            processed_string = processed_string.replace("=", "")
            processed_string = processed_string.replace("+", "")
            processed_string = processed_string.replace("_", " ")
            processed_string = processed_string.replace("^", "")
            processed_string = re.sub(r" {2,}", " ", processed_string)
            processed_string = re.sub(r"\s+", " ", processed_string)
            self.logger.debug("After processed_string: %s", processed_string)
            initial_tokens = [token.strip().lower() for token in processed_string.split(",") if token.strip()]
            self.logger.debug("Initial number of tokens: %s", len(initial_tokens))

            tokens = [self.remove_stop_words(token).strip() for token in initial_tokens]
            self.logger.debug("Number of tokens after removing stop words: %s", len(tokens))

            tokens = [token for token in tokens if 3 <= len(token) <= 75]
            self.logger.debug("Number of tokens after length filtering: %s", len(tokens))

            total_tokens = len(tokens)
            selected_tokens = []
            for token in tokens:
                if total_tokens >= max_length:
                    break
                selected_tokens.append(token)
                total_tokens += 1

            self.logger.debug("Number of selected tokens: %s", len(selected_tokens))

            final_tokens = tokens + selected_tokens
            random.shuffle(final_tokens)

            output_string = ", ".join(token for token in final_tokens if token)
            output_string = output_string.strip(", ")

            self.logger.debug("Final output string: %s", output_string)

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
