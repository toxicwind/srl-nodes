import spacy
import inspect
import textwrap
import nodes
import re
import random
import itertools
from functools import reduce


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

nlp = spacy.load('en_core_web_sm')


class SrlConditionalInterrupt:
    """Interrupt processing if the boolean input is true. Pass through the other input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interrupt": ("BOOLEAN", {"forceInput": True}),
                "inp": (any_typ,),
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output",)
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, interrupt, inp):
        if interrupt:
            nodes.interrupt_processing()

        return (inp,)


class SrlFormatString:
    """Use Python f-string syntax to generate a string using the inputs as the arguments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "first input via str(): {}, second input via repr(): {!r}, third input by index: {2}, fifth input by name: {in4}",
                    },
                ),
            },
            "optional": {
                "in0": (any_typ,),
                "in1": (any_typ,),
                "in2": (any_typ,),
                "in3": (any_typ,),
                "in4": (any_typ,),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, format, **kwargs):
        # Allow referencing arguments both by name and index.
        return (format.format(*kwargs.values(), **kwargs),)


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
                "arg0": (any_typ,),
                "arg1": (any_typ,),
                "arg2": (any_typ,),
                "arg3": (any_typ,),
                "arg4": (any_typ,),
            },
        }

    RETURN_TYPES = (any_typ,)
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
                parameter_names[-len(default_list):], default_list
            )
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
                kw[input_name] for input_name in unnamed_inputs if input_name in kw
            ]

        ret = func(*args)
        return (ret,)


class SrlFilterImageList:
    """Filter an image list based on a list of bools"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "keep": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    def doit(self, images, keep):
        return ([im for im, k in zip(images, keep) if k],)


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
                "input_string": ("STRING", {"multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "doit"
    CATEGORY = "utils"

    def remove_stop_words(self, text):
        """Removes stop words from the input text using spaCy."""
        tokens = text.split(',')
        filtered_tokens = []
        for token in tokens:
            doc = nlp(token)
            filtered_token = ' '.join([t.text for t in doc if not t.is_stop])
            filtered_tokens.append(filtered_token.strip())
        filtered_text = ', '.join(filtered_tokens)
        filtered_text = filtered_text.replace(" (", "(").replace(") ", ")")
        filtered_text = filtered_text.replace("( ", "(").replace(" )", ")")
        filtered_text = filtered_text.replace(" - ", "-")
        return filtered_text

    def doit(self, input_string, seed=0):
        """Cleans, removes stop words, tokenizes, shuffles, and selectively recombines tokens."""
        random.seed(seed)

        preserved_periods = re.sub(
            r"\d+\.(?!\d)", ",", input_string)
        preserved_periods = re.sub(
            r"(?<!\d)\.|\.(?!\d)", ",", preserved_periods)
        preserved_periods = re.sub(
            r"\:(?!\d)", "", preserved_periods)
        preserved_periods = re.sub(r"(?<!\w)'(?!\w)", ",", preserved_periods)
        preserved_periods = re.sub(r"(?<!\w)-(?!\w)", ",", preserved_periods)
        preserved_periods = re.sub(
            r"\((?![^()]*:)[^()]*\)", ",", preserved_periods)
        processed_string = (preserved_periods
                            .replace(";", ",")
                            .replace("!", ",")
                            .replace("?", ",")
                            .replace(" - ", ",")
                            .replace(": ", " ")
                            .replace("“", ",")
                            .replace("”", ",")
                            .replace("‘", ",")
                            .replace("’", ",")
                            .replace("s'", "s")
                            .replace("*", ",")
                            .replace("' ", ",")
                            .replace(" '", ",")
                            .replace("/", "")
                            .replace("\\", "")
                            .replace("[", ",")
                            .replace("]", ",")
                            .replace("{", ",")
                            .replace("}", ",")
                            .replace("<", "")
                            .replace(">", "")
                            .replace("=", "")
                            .replace("+", "")
                            .replace("_", " ")
                            .replace("^", "")
                            .replace("|", ",")
                            .replace(", ", ",")
                            .replace("---", ",")
                            .replace("--", ",")
                            .replace(" - ", ",")
                            .replace("- ", ",")
                            .replace(" -", ",")
                            .replace("…", ",")
                            .replace("=>", ",")
                            .replace("->", ",")
                            .replace("\n", ",")
                            .replace("\"", ",")
                            .replace(",,", ","))

        processed_string = re.sub(r" {2,}", " ", processed_string)
        processed_string = re.sub(r"\s+", " ", processed_string)

        tokens = [
            token.strip().lower()
            for token in processed_string.split(",")
            if token.strip()
        ]

        # Remove stop words for each token and strip afterwards
        tokens = [self.remove_stop_words(token).strip() for token in tokens]

        print(f"Tokens before shuffling: {tokens}")
        shuffled_tokens = random.sample(tokens, len(tokens))
        print(f"Tokens after shuffling: {shuffled_tokens}")
        tokens_with_parentheses = [
            token for token in tokens if '(' in token and ')' in token]
        tokens_without_parentheses = [
            token for token in tokens if token not in tokens_with_parentheses]

        # Ensure there are no spaces after "(" or before ")"
        tokens_with_parentheses = [token.replace(
            " (", "(").replace(") ", ")") for token in tokens_with_parentheses]

        # Shuffle tokens without parentheses
        random.shuffle(tokens_without_parentheses)

        # Calculate the total length of tokens with parentheses
        total_length_with_parentheses = sum(
            len(token) for token in tokens_with_parentheses)

        # Select tokens without parentheses until the total length of all tokens is at least 2048
        selected_tokens_without_parentheses = []
        total_length = total_length_with_parentheses
        for token in tokens_without_parentheses:
            if total_length >= 1804:
                break
            selected_tokens_without_parentheses.append(token)
            total_length += len(token)

        # Combine the two sets of tokens
        final_tokens = tokens_with_parentheses + selected_tokens_without_parentheses
        random.shuffle(final_tokens)

        # Reassemble into a string
        output_string = ', '.join(final_tokens)
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
NODE_CLASS_MAPPINGS["SRL Randomize And Format String"] = SrlRandomizeAndFormatString

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SrlConditionalInterrupt": "SRL Conditional Interrupt",
    "SrlFormatString": "SRL Format String",
    "SrlEval": "SRL Eval",
    "SrlFilterImageList": "SRL Filter Image List",
}
NODE_DISPLAY_NAME_MAPPINGS["SrlRandomizeAndFormatString"] = (
    "SRL Randomize And Format String"
)
