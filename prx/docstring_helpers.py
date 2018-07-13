# ----------------------------------------------------------------------------
# Copyright (c) 2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Docstring helper functions and metaclass."""

import string
import textwrap
import types

__all__ = ('doc_dedent', 'doc_format', 'DocstringSubstituteMeta')


def doc_dedent(doc):
    """Dedent like textwrap.dedent but ignoring unindented first line."""
    # if first line starts with whitespace, just textwrap.dedent
    if not doc or doc[0] in string.whitespace:
        return textwrap.dedent(doc)

    # separate first line from the rest
    lines = doc.split('\n', 1)

    # if only one line, just dedent it
    if len(lines) == 1:
        return textwrap.dedent(doc)

    # multiple lines
    first, rest = lines
    rest = textwrap.dedent(rest)
    return '\n'.join((first, rest))


def doc_format(doc, **kwargs):
    """Format docstring and insert keyword substitutions."""
    # 1) Strip whitespace from substitution strings
    # 2) Dedent so that docstring can match (lack of) indentation in
    # substituted strings
    # 3) Insert substitution strings
    subs = {k: v.strip() for k, v in kwargs.items()}
    return doc_dedent(doc).format(**subs)


class DocstringSubstituteMeta(type):
    """Metaclass for performing docstring substitution from _doc_* attributes.

    This metaclass is useful for building docstrings for classes and methods
    from a class inheritance tree. For classes of this metaclass type, class
    attribute strings starting with '_doc_' can be used to specify
    substitution values to be used in docstrings. The name following the
    '_doc_' prefix is used as the substitution variable name. In addition,
    '_doc_' class attribute strings are inherited from the most recent value
    in classes specified by the '_docstring_bases' list attribute followed by
    all parent classes. Any inherited values can also be used as substitution
    strings in the class's own '_doc_' attribute strings. String substitution
    is carried out for all class and method docstrings using normal
    `string.format` rules.

    Finally, the entire pre-substitution docstring of a parent method or
    classe can be inherited by using '.' as the entire docstring.

    """

    def __new__(cls, name, bases, dct):
        """Create a new class of type DocstringSubstituteMeta."""
        # get docstring substitutions inherited from parent classes
        # and special docstring parents
        parent_substitutions = {}
        parent_docstrings = {}
        for base in reversed(bases):
            for k, v in getattr(base, '_docstring_subs', {}).items():
                parent_substitutions[k] = v
            for k, v in getattr(base, '_docstrings', {}).items():
                parent_docstrings[k] = v
        for base in reversed(dct.get('_docstring_bases', [])):
            for k, v in getattr(base, '_docstring_subs', {}).items():
                parent_substitutions[k] = v

        # get substitutions defined using current class attributes with
        # _doc_ prefix
        self_substitutions = {}
        for k, v in dct.items():
            if k.startswith('_doc_'):
                # normalize whitespace and substitute from parents
                normalized_sub = doc_format(
                    v.lstrip('\n'), **parent_substitutions
                )
                sub_name = k[5:]
                self_substitutions[sub_name] = normalized_sub

        # join self and parent substitutions
        substitutions = parent_substitutions.copy()
        substitutions.update(self_substitutions)

        # format substitution strings with themselves to fill in {{}} values
        # with variables from the current class (don't save these formatted
        # substitutions so that this is re-done with every class)
        subs = {
            k: v.format(**substitutions) for k, v in substitutions.items()
        }

        # get class and method docstrings
        docstrings = {}
        for k, v in dct.items():
            if k == '__doc__':
                # class docstring
                doc = v
            elif isinstance(v, types.FunctionType) and v.__doc__ is not None:
                # method docstring
                doc = v.__doc__
            else:
                continue
            docstrings[k] = doc

        # format docstrings and write to dct
        for k, v in docstrings.items():
            if v == '.':
                # replace with parent docstring
                try:
                    v = parent_docstrings[k]
                except KeyError:
                    errstr = 'Parent docstring for {0} does not exist.'
                    raise ValueError(errstr.format(k))
                # save replaced string so children can use it as well
                docstrings[k] = v
            doc = doc_format(v, **subs)
            if k == '__doc__':
                # class docstring
                dct[k] = doc
            else:
                # method docstring
                # create new function instance so we don't overwrite
                # the docstring of the existing one
                fun_orig = dct[k]
                fun = types.FunctionType(
                    fun_orig.__code__, fun_orig.__globals__, fun_orig.__name__,
                    fun_orig.__defaults__, fun_orig.__closure__,
                )
                fun.__dict__.update(fun_orig.__dict__.copy())
                fun.__doc__ = doc
                dct[k] = fun

        # save all of the class's docstring substitutions as an attribute
        # that can be inspected and used by subclasses as above
        dct['_docstring_subs'] = substitutions
        # save all of the unformatted docstrings for this class and parents
        # to use for '.' substitution
        dct['_docstrings'] = parent_docstrings.copy()
        dct['_docstrings'].update(docstrings)

        return super(DocstringSubstituteMeta, cls).__new__(
            cls, name, bases, dct,
        )
